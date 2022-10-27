#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn
import tqdm

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import DEFAULT_SELECTOR, Tensorizer, MultiSetDataIterator
from dpr.utils.model_utils import CheckpointState, move_to_device

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiEncoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
        "query_absolute_idxs", 
        "positive_passage_absolute_idxs",
        "negative_passage_absolute_idxs",
        "hard_negative_passage_absolute_idxs",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)

def _print_tensors():
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # r = torch.matmul(q_vectors, ctx_vectors.T)
    # return r
    return torch.matmul(q_vectors, ctx_vectors.T)


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class CoordinateAscentStatus:
    DISABLED = 0
    TRAIN_Q = 1
    TRAIN_CTX = 2


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""
    _most_recent_num_correct: Dict[CoordinateAscentStatus, int]
    _ctx_idx_to_query_idx: Dict[int, Set[int]]
    _query_idx_to_ctx_idx: Dict[int, Set[int]]

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
        coordinate_ascent_status: bool = CoordinateAscentStatus.DISABLED
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        # *** variables used for coordinate ascent ***
        self.coordinate_ascent_status = coordinate_ascent_status
        self.stored_q_vectors = None
        self.stored_q_idxs = None
        self.stored_ctx_vectors = None
        self.stored_ctx_idxs = None
        self._most_recent_num_correct = { CoordinateAscentStatus.TRAIN_Q: 0, CoordinateAscentStatus.TRAIN_CTX: 0 }
        # During precomputing, store the absolute indices of the precomputed stuff, so we can determine
        # if there are multiple things associated with it later.
        self._pos_ctx_idxs = None
        self._query_idxs = None
        self._num_positive_samples = None

        # For each passage index in the dataset, we store the indices of associated queries.
        self._ctx_idx_to_query_idx = collections.defaultdict(set)
        self._query_idx_to_ctx_idx = collections.defaultdict(set)

        self.use_min_criteria_for_toggle = None # will be overriden in train_...

        # index of stored negative passages corresponding to each stored positive passage.
        self._stored_negative_passage_idxs_by_query_idx = None # will be overriden each time we get positive passage

        ############################ IDF Statistics ##############################
        self.use_idf_encoder = True

        self._embedding_dim = 768
        self._vocab_size = 30522

        bottleneck_embedding_dim = 128
        self.idf_sparse_embed = nn.Sequential(
            nn.Linear(self._vocab_size, bottleneck_embedding_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_embedding_dim, self._embedding_dim),
            nn.ReLU(),
        )
        self.joint_idf_embed = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._embedding_dim),
            nn.ReLU(),
            nn.Linear(self._embedding_dim, self._embedding_dim)
        )
        self.reset_idf()
    
    def process_batch_idf(self, batch: BiEncoderBatch) -> None:
        input_ids = batch.context_ids.flatten()
        self._IDF_corpus_size += len(input_ids)
        self._IDF_frequencies += input_ids.bincount(minlength=self._vocab_size)
        
    def reset_idf(self) -> None:
        self._IDF_corpus_size = 0
        self._IDF_frequencies = torch.zeros((self._vocab_size,), dtype=int, requires_grad=False) 
    
    def _get_positive_passage_idxs(self, train_iterator: MultiSetDataIterator, shuffle_positives: bool) -> Set[int]:
        """
        'What body of water does the Colorado River flow into?' and 
        'What body of water does the Colorado River empty into?'

        are both in TREC...
        """
        if not len(self._ctx_idx_to_query_idx):
            print("Getting indices of all positive passages...(we only do this once)...")
            data = train_iterator.iterate_ds_data(epoch=0)
            i = 0
            idx_to_sample = {}
            positive_ctx_idx_to_query_idxs = collections.defaultdict(set)
            for batch, _dataset_idx in data:
                for sample in batch:
                    idx_to_sample[sample.query_idx] = sample
                    if shuffle_positives:
                        positive_passages = sample.positive_passages
                    else:
                        positive_passages = [sample.positive_passages[0]]
                    
                    for positive_passage in positive_passages:
                        positive_ctx_idx_to_query_idxs[positive_passage.index].add(i)
                        self._ctx_idx_to_query_idx[positive_passage.index].add(sample.query_idx)
                        self._query_idx_to_ctx_idx[sample.query_idx].add(positive_passage.index)

            # import pdb; pdb.set_trace()
            # 
            # overlap in positive passages is bad.
            # determined that 355 / 1127 (30.8%) of queries in TREC have a different query with a shared
            # positive. maybe this isn't a big deal in the contrastive loss (although it probably is) but
            # it is a HUGE deal for full softmax since that means literally 30.8% of the time the model has to
            # resort to random guessing.
            # 
            # but on the other hand, only 898/15190 (5.9%) of positive passages in TREC are associated
            # with more than one thing.
            # 
            # there are also some really noisy things in this dataset:
            # (Pdb) p idx_to_sample[418].query
            # 
            # 'How many people does Honda employ in the U.S.?'
            # (Pdb) p idx_to_sample[418].positive_passages[0]
            # 
            # BiEncoderPassage(text='words \'How R U Doin?\' appearing on-screen one last time. How R U Doin? "How R U Doin?" is a song by Danish-Norwegian pop group Aqua from their third studio album, "Megalomania". It was released as the album\'s lead single on 14 March 2011. The song peaked at number four in Denmark, becoming the group\'s tenth top-ten single. It has since been certified gold by the International Federation of the Phonographic Industry (IFPI) for sales of 15,000 copies. On 9 March 2011 Aqua posted the single artwork to "How R U Doin?" on their Facebook page with the date "14.03.2011".', title='How R U Doin?', index=15429849)
            # (Pdb) p idx_to_sample[418].positive_passages[1]
            # BiEncoderPassage(text='How R U Doin? "How R U Doin?" is a song by Danish-Norwegian pop group Aqua from their third studio album, "Megalomania". It was released as the album\'s lead single on 14 March 2011. The song peaked at number four in Denmark, becoming the group\'s tenth top-ten single. It has since been certified gold by the International Federation of the Phonographic Industry (IFPI) for sales of 15,000 copies. On 9 March 2011 Aqua posted the single artwork to "How R U Doin?" on their Facebook page with the date "14.03.2011". The Europop song was written by Søren Rasted, Claus Norreen,', title='How R U Doin?', index=15429847)
            # 

            # store equivalent idxs for each thing in the dataset. 
            multipassages = {k: v for k,v in self._ctx_idx_to_query_idx.items() if len(v) > 1}

        return self._ctx_idx_to_query_idx.keys()

    
    def _precompute_embeddings_full(
            self, cfg, ds_cfg, tensorizer, train_iterator: MultiSetDataIterator,
            num_hard_negatives: int
        ) -> Tuple[List[Optional[T]], List[Optional[T]]]:
        # print("precompute embeddings - printing tensors:")
        # _print_tensors()
        torch.cuda.empty_cache() 
        query_vectors = []
        pos_ctxs = []
        neg_ctxs = []

        train_iterator.shuffle = False

        # ((TREC) Question Classification dataset contains 5500 labeled questions in training set and 
        # another 500 for test set.)
        print(f"precomputing embeddings with {num_hard_negatives} hard negatives")
        
        data = train_iterator.iterate_ds_data(epoch=0)
        query_idxs = []
        positive_passage_idxs = []
        negative_passage_idxs = []
        already_seen_idxs = set()
        all_positive_passage_idxs = self._get_positive_passage_idxs(
            train_iterator=train_iterator, shuffle_positives=cfg.train.shuffle_positives_override
        )

        print("TODO: consider using 'regular' negatives as well in training.")

        assert len(ds_cfg.train_datasets) == 1
        ds_cfg = ds_cfg.train_datasets[0]

        self._stored_negative_passage_idxs_by_query_idx = collections.defaultdict(list)

        num_stored_negatives = 0 
        self._num_positive_samples = 0
        for i, samples_batch in tqdm.tqdm(
            enumerate(data),
            colour="red", leave=False,
            desc=f"Precomputing embeddings with hn={num_hard_negatives}",
            total=train_iterator.get_max_iterations()
        ):

            # print(f"\tprecomputing step {i} memory usage - {torch.cuda.memory_allocated()} / {torch.cuda.max_memory_allocated()}")
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            special_token = ds_cfg.special_token
            encoder_type = ds_cfg.encoder_type
            shuffle_positives = cfg.train.shuffle_positives_override

            for sample in samples_batch:
                self._num_positive_samples += 1
                # Filter out stuff we've already seen.
                sample.hard_negative_passages = [
                    hn for hn in sample.hard_negative_passages
                    if ((hn.index not in already_seen_idxs) and (hn.index not in all_positive_passage_idxs))
                ]
                # Shuffle so we can get different hard negatives every epoch.
                random.shuffle(sample.hard_negative_passages)
                sample.hard_negative_passages = (
                    sample.hard_negative_passages[:num_hard_negatives]
                ) 
                for hn in sample.hard_negative_passages:
                    already_seen_idxs.add(hn.index)
                    self._stored_negative_passage_idxs_by_query_idx[sample.query_idx].append(
                        num_stored_negatives
                    )
                    num_stored_negatives += 1

            biencoder_input = self.create_biencoder_input(
                samples=samples_batch,
                tensorizer=tensorizer,
                insert_title=True,
                num_hard_negatives=num_hard_negatives, # set to 100 to use them all!
                num_other_negatives=0,
                shuffle=False,
                shuffle_positives=shuffle_positives,
                hard_neg_fallback=False,
                query_token=special_token,
            )

            model_device = next(self.parameters()).device

            biencoder_input = BiEncoderBatch(
                **move_to_device(biencoder_input._asdict(), model_device)
            )

            # get the token to be used for representation selection
            selector = ds_cfg.selector if ds_cfg else DEFAULT_SELECTOR

            rep_positions = selector.get_positions(biencoder_input.question_ids, tensorizer)

            q_attn_mask = tensorizer.get_attn_mask(
                biencoder_input.question_ids
            )
            ctx_attn_mask = tensorizer.get_attn_mask(
                biencoder_input.context_ids
            )

            # Flip coordinate ascent status so we get embedding
            # for the thing we won't be training during this epoch.
            coordinate_ascent_status = (
                CoordinateAscentStatus.TRAIN_Q
                if self.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_CTX else 
                CoordinateAscentStatus.TRAIN_CTX
            )
            ## TODO: filter out so we don't need to get
            ## vectors for negative passages that have an idx
            ## that's already stored in negative_passage_idxs.
            with torch.no_grad():
                local_q_vector, local_ctx_vectors = self(
                    biencoder_input.question_ids,
                    biencoder_input.question_segments,
                    q_attn_mask,
                    biencoder_input.context_ids,
                    biencoder_input.ctx_segments,
                    ctx_attn_mask,
                    encoder_type=encoder_type,
                    representation_token_pos=rep_positions,
                    coordinate_ascent_status=coordinate_ascent_status
                )

            # TODO: make this block of code more idiomatic pytorch.
            # Collect context vectors.
            if (local_ctx_vectors is not None) and len(local_ctx_vectors) > 0:
                # Track indices of all the passage idxs.
                new_positive_passage_idxs = biencoder_input.positive_passage_absolute_idxs
                positive_passage_idxs.extend(new_positive_passage_idxs)

                new_negative_passage_idxs = [
                    idx 
                    for idxs in biencoder_input.hard_negative_passage_absolute_idxs 
                    for idx in idxs[:num_hard_negatives].tolist() 
                ]
                negative_passage_idxs.extend(new_negative_passage_idxs)
                # Split local_ctx_vectors into positive and negative contexts.
                ctx_mask = torch.zeros(len(local_ctx_vectors), dtype=torch.bool)
                ctx_mask.scatter_(0, torch.tensor(biencoder_input.is_positive), 1)
                ctx_mask = ctx_mask.to(local_ctx_vectors.device)
                new_pos_ctxs = local_ctx_vectors.masked_select(ctx_mask[:, None]).reshape(-1, 768)
                pos_ctxs.append(new_pos_ctxs.cpu())

                new_neg_ctxs = local_ctx_vectors.masked_select(~ctx_mask[:, None]).reshape(-1, 768)
                neg_ctxs.append(new_neg_ctxs.cpu())

                if len(new_negative_passage_idxs) != len(new_neg_ctxs):
                    import pdb; pdb.set_trace()
                    raise RuntimeError(f'got {new_negative_passage_idxs} idxs and {new_neg_ctxs} embeddings')
            else:
                # Otherwise, we're precomputing queries - collect query vectors and indices.
                assert (local_q_vector is not None), "need local_q_vector because we're precomputing queries"
                query_vectors.append(local_q_vector.cpu())
                query_idxs.append(biencoder_input.query_absolute_idxs.cpu())

            # if i == train_iterator.get_max_iterations()-1: import pdb; pdb.set_trace()

        print("Done precomputing - computing overlaps...")

        if len(query_vectors) > 0:
            query_vectors = torch.cat(query_vectors, dim=0)
            all_ctxs = []
            # Store indices for queries.
            self._pos_ctx_idxs = None
            self._query_idxs = torch.cat(query_idxs).cpu()
        else:
            # Ensure no negative contexts that have IDs that appear in positive contexts.
            positive_passage_idxs = torch.tensor(positive_passage_idxs)
            if len(negative_passage_idxs) > 0:
                negative_passage_idxs = torch.tensor(negative_passage_idxs)
                negative_ctxs_duplicate_mask = (
                    negative_passage_idxs[:, None] == positive_passage_idxs[None, :]
                ).any(dim=1)
                assert negative_ctxs_duplicate_mask.sum() == 0, "found negative contexts that were already positive contexts"
                
            all_ctxs = torch.cat((pos_ctxs + neg_ctxs), dim=0)
            # Store indices for positive contexts. This lets us figure out
            # *all* the queries that correspond to it later.
            self._pos_ctx_idxs = positive_passage_idxs
            self._query_idxs = None

        return query_vectors, all_ctxs
    
    def _precompute_embeddings_ctx(self, cfg, ds_cfg, tensorizer, train_iterator: MultiSetDataIterator, num_hard_negatives: int) -> None:
        print(f"precomputing ctx embeddings with {num_hard_negatives} hard negatives. (how do we actually do this for all possible contexts??)")
        q_emb_list, ctx_emb_list = self._precompute_embeddings_full(
            cfg=cfg,
            ds_cfg=ds_cfg,
            tensorizer=tensorizer,
            train_iterator=train_iterator,
            num_hard_negatives=num_hard_negatives
        )
        assert len(q_emb_list) == 0
        # ctx_emb_list should have been concatenated when duplicates were filtered in the precompute step.
        print(f"precomputed {len(ctx_emb_list)} context embeddings")
        return ctx_emb_list.cuda()
    
    def _precompute_embeddings_q(self, cfg, ds_cfg, tensorizer, train_iterator: MultiSetDataIterator) -> None:
        q_emb_list, ctx_emb_list = self._precompute_embeddings_full(
            cfg=cfg,
            ds_cfg=ds_cfg,
            tensorizer=tensorizer,
            train_iterator=train_iterator,
            num_hard_negatives=0 # no hard negatives needed if we're just encoding the queries
        )
        assert len(ctx_emb_list) == 0
        print(f"precomputed {len(q_emb_list)} query embeddings")

        return q_emb_list.cuda()
    
    def pre_epoch(self, cfg, ds_cfg, tensorizer, train_iterator: MultiSetDataIterator, num_hard_negatives: int) -> None:
        print("BiEncoder pre_epoch() called")
        print(f"pre_epoch memory usage [before precomputing] - {torch.cuda.memory_allocated()} / {torch.cuda.max_memory_allocated()}")
        #
        #  precompute embeddings
        #
        assert self.training # make sure we're not in eval mode
        if self.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_CTX:
            if self.stored_ctx_vectors is not None:
                self.stored_ctx_vectors = self.stored_ctx_vectors.cpu()
            self.stored_ctx_vectors = None
            self.stored_q_vectors = self._precompute_embeddings_q(
                cfg=cfg,
                ds_cfg=ds_cfg,
                tensorizer=tensorizer,
                train_iterator=train_iterator
            )
            print("self.stored_q_vectors.shape =", self.stored_q_vectors.shape)
        elif self.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_Q:
            if self.stored_q_vectors is not None:
                self.stored_q_vectors = self.stored_q_vectors.cpu()
            self.stored_q_vectors = None
            self.stored_ctx_vectors = self._precompute_embeddings_ctx(
                cfg=cfg,
                ds_cfg=ds_cfg,
                tensorizer=tensorizer,
                train_iterator=train_iterator,
                num_hard_negatives=num_hard_negatives
            )
            print("self.stored_ctx_vectors.shape =", self.stored_ctx_vectors.shape)
        else: # coordinate ascent disabled - do nothing
            pass
        
        # flip switch **after** precomputing embeddings, so that we know
        # we were still in the prev mode so that the model returned None
        # for the other type.
        # self._toggle_ca_status()
        print(f"pre_epoch memory usage [after precomputing] - {torch.cuda.memory_allocated()} / {torch.cuda.max_memory_allocated()}")
        print("pre_epoch self.coordinate_ascent_status =", self.coordinate_ascent_status)

    def post_epoch(self, num_correct: int) -> CoordinateAscentStatus:
        """Toggles status based on num_correct. Sets status to the status which most recently
        did worse on the training examples.

        Args:
            num_correct (int): number of correct predictions in the most recent epoch
        Returns:
            status (CoordinateAscentStatus): status of the encoder for the next epoch
        """
        print(f"post_epoch memory usage - {torch.cuda.memory_allocated()} / {torch.cuda.max_memory_allocated()}")
        if self.use_min_criteria_for_toggle:
            self._most_recent_num_correct[self.coordinate_ascent_status] = num_correct
            self.coordinate_ascent_status = min(self._most_recent_num_correct, key=self._most_recent_num_correct.get)
        else:
            self._toggle_ca_status()

    def _toggle_ca_status(self) -> None:
        """Toggles status."""
        print("BiEncoder post_epoch() called")
        # 
        #   advance coordinate ascent status
        #
        if self.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_Q:
            #    training query encoder - train ctx next
            self.coordinate_ascent_status = CoordinateAscentStatus.TRAIN_CTX
        elif self.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_CTX:
            #    training context encoder - train query next
            self.coordinate_ascent_status = CoordinateAscentStatus.TRAIN_Q
        else:
            #    coordinate ascent disabled - do nothing
            pass        
    
    @property
    def _idf(self) -> torch.Tensor:
        assert (self._IDF_corpus_size > 0), "can't compute IDF without any documents!"
        return torch.log(self._IDF_corpus_size - self._IDF_frequencies + 0.5) - torch.log(self._IDF_frequencies + 0.5)
    
    def idf_embed_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Augments word embeddings with IDF information."""
        batch_size, sequence_length, _ = token_embeddings.shape
        idf_vector = self._idf.to(token_embeddings.device)
        idf_embedding = self.idf_sparse_embed(idf_vector[None])
        assert idf_embedding.shape == (1, 768)
        idf_embedding = idf_embedding[None].repeat((batch_size, sequence_length, 1))
        ################################################################################
        idf_embedding = idf_embedding.reshape((batch_size * sequence_length, self._embedding_dim))
        token_embeddings = idf_embedding.reshape((batch_size * sequence_length, self._embedding_dim))
        joint_embedding = torch.cat((idf_embedding, token_embeddings), dim=1)
        assert joint_embedding.shape == (batch_size * sequence_length, self._embedding_dim * 2)
        ################################################################################
        out_embedding = self.joint_idf_embed(joint_embedding)
        assert out_embedding.shape == (batch_size * sequence_length, self._embedding_dim)
        return out_embedding.reshape((batch_size, sequence_length, self._embedding_dim))

    def get_representation(
        self,
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        """Gets context representation and optionally adds IDF."""
        if ids is None:
            return (None, None, None)
        token_embeddings = sub_model.embeddings.word_embeddings.forward(ids)

        if self.use_idf_encoder:
            gamma = 0.1 # hparm
            token_embeddings += (gamma * self.idf_embed_tokens(token_embeddings))

        sequence_output, pooled_output, hidden_states = sub_model(
            attention_mask=attn_mask,
            token_type_ids=segments,
            representation_token_pos=representation_token_pos,
            inputs_embeds=token_embeddings,
        )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
        coordinate_ascent_status: Optional[CoordinateAscentStatus] = None,
    ) -> Tuple[T, T]:
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model

        coordinate_ascent_status = (coordinate_ascent_status or self.coordinate_ascent_status)

        if coordinate_ascent_status == CoordinateAscentStatus.TRAIN_CTX:
            q_pooled_out = None
        else:
            _q_seq, q_pooled_out, _q_hidden = self.get_representation(
                q_encoder,
                question_ids,
                question_segments,
                question_attn_mask,
                fix_encoder=self.fix_q_encoder,
                representation_token_pos=representation_token_pos,
            )

        if coordinate_ascent_status == CoordinateAscentStatus.TRAIN_Q:
            ctx_pooled_out = None
        else:
            ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
            _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
                ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, fix_encoder=self.fix_ctx_encoder
            )

        return q_pooled_out, ctx_pooled_out

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        query_absolute_idxs = []
        positive_passage_absolute_idxs = []
        negative_passage_absolute_idxs = []
        hard_negative_passage_absolute_idxs = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            ########################################################
            query_absolute_idxs.append(sample.query_idx)
            positive_passage_absolute_idxs.append(
                positive_ctx.index
            )
            negative_passage_absolute_idxs.append(
                torch.tensor([p.index for p in sample.negative_passages])
            )
            hard_negative_passage_absolute_idxs.append(
                torch.tensor([p.index for p in sample.hard_negative_passages])
            )
            ########################################################

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)


        query_absolute_idxs = torch.tensor(query_absolute_idxs)
        # can't stack tensors because they may be different lengths
        # negative_passage_absolute_idxs = torch.stack(
        #     negative_passage_absolute_idxs
        # )
        # hard_negative_passage_absolute_idxs = torch.stack(
        #     hard_negative_passage_absolute_idxs
        # )

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
            query_absolute_idxs,
            positive_passage_absolute_idxs,
            negative_passage_absolute_idxs,
            hard_negative_passage_absolute_idxs,
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)
        print("*** Loaded state dict ***")

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):
    biencoder: Optional[BiEncoder]
    def __init__(self, biencoder: Optional[BiEncoder] = None, use_full_softmax: bool = False):
        self.biencoder = biencoder
        self.use_full_softmax = use_full_softmax
    
    def _get_true_mask_for_query(self, query_absolute_idxs) -> T:
        assert self.biencoder._pos_ctx_idxs is not None
        mask = []
        for query_idx in query_absolute_idxs.tolist():
            ctx_idxs = list(self.biencoder._query_idx_to_ctx_idx[query_idx])
            assert len(ctx_idxs) > 0, f"got 0 contexts for query idx {query_idx}"
            mask.append(
                (
                    self.biencoder._pos_ctx_idxs[:, None] == torch.tensor(ctx_idxs)[None, :]
                ).any(dim=1)
            )
        out_mask = torch.stack(mask).cuda()
        assert (out_mask.sum(1) > 0).all()
        return out_mask
    
    def _get_true_mask_for_context(self, context_absolute_idxs) -> T:
        assert self.biencoder._query_idxs is not None
        mask = []
        for context_idx in context_absolute_idxs:
            query_idxs = list(self.biencoder._ctx_idx_to_query_idx[context_idx])
            assert len(query_idxs) > 0, f"got 0 queries for context idx {context_idx}"
            mask.append(
                (
                    self.biencoder._query_idxs[:, None] == torch.tensor(query_idxs)[None, :]
                ).any(dim=1)
            )
        out_mask = torch.stack(mask).cuda()
        assert (out_mask.sum(1) > 0).all()
        return out_mask

    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        hard_negative_idx_per_question: list = None,
        query_absolute_idxs: Optional[T] = None,
        passage_absolute_idxs: Optional[T] = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """

        if (self.biencoder is None) or (self.biencoder.coordinate_ascent_status == CoordinateAscentStatus.DISABLED):  # regular contrastive loss
            loss, correct_predictions_count = self._calc_contrastive_loss(
                q_vectors=q_vectors,
                ctx_vectors=ctx_vectors,
                positive_idx_per_question=positive_idx_per_question,
            )
        else:
            if self.biencoder.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_Q:
                loss, correct_predictions_count = self._calc_ca_loss(
                    batch_vectors=q_vectors,
                    stored_vectors=self.biencoder.stored_ctx_vectors,
                    true_idx_mask=self._get_true_mask_for_query(query_absolute_idxs),
                    absolute_idxs=query_absolute_idxs
                )
            else:
                assert self.biencoder.coordinate_ascent_status == CoordinateAscentStatus.TRAIN_CTX
                loss, correct_predictions_count = self._calc_ca_loss(
                    batch_vectors=ctx_vectors,
                    stored_vectors=self.biencoder.stored_q_vectors,
                    true_idx_mask=self._get_true_mask_for_context(passage_absolute_idxs),
                    absolute_idxs=query_absolute_idxs,
                )

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count
    
    def _calc_ca_loss_train_query(
            self,
            batch_vectors: T,
            stored_vectors: T,
            true_idx_mask: T,
            absolute_idxs: T,
            loss_scale: float = None
        ) -> Tuple[T, int]:
        """Computes full softmax over all `stored_vectors`. When training the query encoder,
        we can use all of the precomputed samples as negative passages.
        """

        assert stored_vectors is not None, f"got None stored_vectors with coordinate_ascent_status {self.biencoder.coordinate_ascent_status}"

        if not self.use_full_softmax:
            # 
            # When USE_FULL_SOFTMAX is false, we only compute loss for positive
            # + negative samples *in the batch*.
            # 

            # Get indices of all negative samples so we can include them in the loss too
            neg_idxs = []
            for idx in absolute_idxs.cpu().tolist():
                neg_idxs.extend(self.biencoder._stored_negative_passage_idxs_by_query_idx[idx])
            if len(neg_idxs) > 0:
                neg_idxs = torch.tensor(neg_idxs) + self.biencoder._num_positive_samples
                all_sample_idxs = torch.cat((absolute_idxs, neg_idxs.to(absolute_idxs.device)), dim=0)
            else:
                all_sample_idxs = absolute_idxs
            true_idx_mask = true_idx_mask[:, absolute_idxs]
            
            # stored_vectors = stored_vectors[all_sample_idxs]
        
        else:
            pass
            ########################################################################
            ########  Following lines samples from only within the batch   #########
            ########################################################################
            # When use_full_softmax is true, we still only use negative samples from within the batch.
            # neg_idxs = []
            # for idx in absolute_idxs.cpu().tolist():
            #     neg_idxs.extend(self.biencoder._stored_negative_passage_idxs_by_query_idx[idx])
            # if len(neg_idxs) > 0:
            #     neg_idxs = torch.tensor(neg_idxs) + self.biencoder._num_positive_samples
            #     all_sample_idxs = torch.cat(
            #         (torch.arange(self.biencoder._num_positive_samples), neg_idxs), dim=0
            #     ).to(absolute_idxs.device)
            # else:
            #     all_sample_idxs = torch.arange(self.biencoder._num_positive_samples).to(absolute_idxs.device)
            # import pdb; pdb.set_trace()
            # stored_vectors = stored_vectors[all_sample_idxs]

        # import pdb; pdb.set_trace()
        # print('indexing the thingy. true_idx_mask.shape:', true_idx_mask.shape, 'absolute_idxs.shape:', absolute_idxs.shape)
        # print('absolute_idxs:', absolute_idxs.cpu().tolist())
        
        sims = self.get_scores(batch_vectors, stored_vectors)
        # softmax_scores = sims.log_softmax(dim=1)
        # import pdb; pdb.set_trace()

        # We need to normalize the labels so that they
        # are on the simplex, i.e. they look like
        # [0.25, 0, 0, 0.25, 0.25, 0.25] instead of
        # [1, 0, 0, 1, 1, 1].
        positive_labels = (true_idx_mask.float() / true_idx_mask.sum(1)[:, None].float()).float()

        batch_size = sims.shape[0]
        num_neg_ctxs = sims.shape[1] - positive_labels.shape[1]
        negative_labels = torch.zeros(batch_size, num_neg_ctxs).to(positive_labels.device)

        all_labels = torch.cat(
            (positive_labels, negative_labels), dim=1
        )
        assert all_labels.shape == sims.shape

        loss = nn.CrossEntropyLoss()(sims, all_labels)
        # print("absolute_idxs:", absolute_idxs.tolist())
        # print("softmax_scores.shape:", softmax_scores.shape, "absolute_idxs.shape:", absolute_idxs.shape)
        # loss = F.nll_loss(
        #     softmax_scores,
        #     query_absolute_idxs.to(softmax_scores.device),
        #     reduction="mean",
        # )


        # todo: how to compute this w/ multiple correct idxs?
        _max_score, max_idxs = torch.max(sims, 1)

        if self.use_full_softmax:
            correct_predictions_count = (
                max_idxs == absolute_idxs.to(max_idxs.device)
            ).sum()
        else:
            correct_predictions_count = (
                max_idxs == torch.arange(batch_size).to(max_idxs.device)
            ).sum()

        return loss, correct_predictions_count

    
    def _calc_contrastive_loss(
            self,
            q_vectors: T,
            ctx_vectors: T,
            positive_idx_per_question: list,
            hard_negative_idx_per_question: list = None,
        ) -> Tuple[T, int]:
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        _max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
