# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import math
import torch
import contextlib
from typing import List, Tuple
import torch.nn as nn

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules import (
    PositionalEmbedding,
    Fp32GroupNorm,
    FairseqDropout,
    SamePad,
    GradMultiply,
    LayerNorm,
    Fp32LayerNorm,
    TransposeLast,
)
import numpy as np

from .encoder import TransformerEncoder
import copy

import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding

logger = logging.getLogger(__name__)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from fairseq.data import data_utils

def collate(
    source,
    unmasked_source,
    targets,
    pad_idx,
    eos_idx,
    task_name = 'speech_text_pretrain',
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(source) == 0:
        return {}
    
    # print('before collating')
    # print(source)
    # print(targets)

    no_boseos_targets = [s[1:-1] for s in targets]

        
    def merge(input, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            input,
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    if task_name == 's2t':
        src_tokens = data_utils.collate_tokens(
            source,
            pad_idx,
            eos_idx,
            left_pad=False,
            move_eos_to_beginning=False,
        )
        src_lengths = (~src_tokens.eq(pad_idx)).sum(dim = -1)
    else:
        src_tokens = merge(
            source,
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        src_lengths = torch.LongTensor([s.numel() for s in source])
    if len(unmasked_source) != 0:
        unmasked_src_tokens = merge(
            unmasked_source,
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
    else:
        unmasked_src_tokens = None

    prev_output_tokens = None
    target = None
    target = merge(
        targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )

    no_boseos_target = merge(
        no_boseos_targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    ntokens = sum([len(cur_target) for cur_target in targets])

    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            targets,
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        
    # print('after collating')
    # print(src_tokens)  
    # print(prev_output_tokens)
    # print(targets)
        
    # print([len(s) for s in src_tokens])
    # print([len(s) for s in unmasked_src_tokens])
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "target_list": [unmasked_src_tokens]
        },
        "target": target,
        "no_boseos_target": no_boseos_target,
        "nsentences": len(source),
        "task_name": task_name,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch



def collate_onehot(
    source,
    unmasked_source,
    targets,
    pad_idx,
    task_name = 'speech_text_pretrain',
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(source) == 0:
        return {}
    
    def merge(input, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            input,
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )
        
        
    def merge_onehot(input):
        sizes = [len(s) for s in input]
        target_size = max(sizes)
        
        input_tokens = input[0].new_zeros(
            len(input), target_size, input[0].size(-1)
        )
        padding_mask = torch.BoolTensor(input_tokens.shape[:-1]).fill_(False).to(input[0].device)
        for i, (f, size) in enumerate(zip(input, sizes)):
            input_tokens[i, :size] = f[:size]
            input_tokens[i, size:, pad_idx] = 1
            padding_mask[i, size:] = True

        input_lengths = torch.LongTensor(sizes).to(input[0].device)

        return input_tokens, padding_mask, input_lengths
        

    no_boseos_targets = [s[1:-1] for s in targets]
    
    src_tokens, src_padding_mask, src_lengths = merge_onehot(source)
    # print(src_tokens[0].argmax(-1), src_tokens.size(), src_padding_mask[0], src_padding_mask[0].sum(), src_lengths[0])
    if len(unmasked_source) != 0:
        unmasked_src_tokens, unmasked_src_padding_mask, unmasked_src_lengths = merge_onehot(unmasked_source)
    else:
        unmasked_src_tokens = None
        unmasked_src_padding_mask = None
        unmasked_src_lengths = None
        
    prev_output_tokens = None
    target = None
    target = merge(
        targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )

    no_boseos_target = merge(
        no_boseos_targets,
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    ntokens = sum([len(cur_target) for cur_target in targets])

    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            targets,
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        
    # print('after collating')
    # print(src_tokens)  
    # print(prev_output_tokens)
    # print(targets)
        
    # print([len(s) for s in src_tokens])
    # print([len(s) for s in unmasked_src_tokens])
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_padding_mask": src_padding_mask,
            "unmasked_src_lengths": unmasked_src_lengths,
            "unmasked_src_padding_mask": unmasked_src_padding_mask,
            "target_list": [unmasked_src_tokens]
        },
        "target": target,
        "no_boseos_target": no_boseos_target,
        "nsentences": len(source),
        "task_name": task_name,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


def fake_collate(audio_tokens, task_name = 'speech_text_pretrain'):
    ntokens = sum([len(cur_target) for cur_target in audio_tokens])
    src_lengths = torch.LongTensor([s.numel() for s in audio_tokens])
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": audio_tokens,
            "src_lengths": src_lengths,
        },
        "nsentences": len(audio_tokens),
        "task_name": task_name,
    }

    return batch


def collater_audio(features):
    sizes = [len(s) for s in features]
    # print('audio', sizes)
    target_size = max(sizes)

    collated_features = features[0].new_zeros(
        len(features), target_size, features[0].size(-1)
    )
    padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
    for i, (f, size) in enumerate(zip(features, sizes)):
        real_size = size
        collated_features[i, :real_size] = f[:real_size]
        padding_mask[i, size:] = True

    return collated_features, padding_mask, target_size



class KmeansVectorQuantizer(nn.Module):
    def __init__(
        self, dim, num_vars, nspecial= 4, init_clus_dir=None, 
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.input_dim = dim
        self.num_vars = num_vars
        self.embedding = nn.Parameter(
            0.01 * torch.randn(num_vars, self.input_dim)
        )
        self.nspecial = nspecial
        if init_clus_dir is not None:
            self.embedding.data = torch.from_numpy(np.load(init_clus_dir))
            # self.embedding.data[:self.nspecial, :] = float("inf")
            # self.embedding.data[-1, :] = float("inf")
            logger.info(f'Loaded cluster centroids from {init_clus_dir}')
    # x is of shape TxC
    def forward(self, x, sampling= False):
        #print(x.size(), torch.sum(x**2, dim=-1, keepdim=True).size(), torch.sum(self.embedding**2, dim=1).unsqueeze(0).size(), torch.matmul(x, self.embedding.t()).size())

        # self.embedding.data[:self.nspecial, :] = float("inf")
        # self.embedding.data[-1, :] = float("inf")
        self.embedding.requires_grad = False
        distances = (torch.sum(x**2, dim=-1, keepdim=True) 
                        + torch.sum(self.embedding**2, dim=1).unsqueeze(0)
                        - 2 * torch.matmul(x, self.embedding.t()))
        lprobs = F.softmax(-distances, dim=-1)
        # lprobs[:, :self.nspecial] = lprobs[:, :self.nspecial] * 0
        # lprobs[:, -1] = lprobs[:, -1] * 0
        # lprobs = lprobs / (1e-7 + lprobs.sum(dim = -1, keepdim = True))
        # print(distances.size())
        quantized_x = torch.matmul(lprobs, self.embedding)
        dictionary_loss = (((x.detach() - quantized_x) ** 2).sum()) / x.size(0)
        commitment_loss = (((x - quantized_x.detach()) ** 2).sum()) / x.size(0)
        
        if not sampling:
            idx = torch.max(lprobs, dim=-1)[1]
        else:
            pred = torch.argmax(lprobs, dim=-1, keepdim= True)
            idx = torch.zeros_like(lprobs).scatter_(-1, pred, 1)
            # print('-2', pred, idx.size(),idx.argmax(-1))
            idx_soft = lprobs
            idx = (idx - idx_soft).detach() + idx_soft
            # idx = idx_soft
        
        return idx, lprobs, dictionary_loss+commitment_loss

class LinearLayer(nn.Module):
    def __init__(self, idim, odom, dropout=0):
        super(LinearLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(idim, odom),
            nn.LayerNorm(odom),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        return out
    
    def forward(self, src_tokens, src_lengths):
        """
        src_tokens: [B, T, C]
        src_lengths: [B]
        """
        x = self.linear(src_tokens)
        x = x.transpose(0, 1).contiguous() # -> T x B x C
        return x, src_lengths


class TdnnSegmenter(nn.Module):
    def __init__(self, num_features, out_dim):
        super().__init__()
        # self.tdnn = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=num_features,
        #         out_channels=500,
        #         kernel_size=11,
        #         stride=1,
        #         padding=5,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(
        #         in_channels=500,
        #         out_channels=500,
        #         kernel_size=7,
        #         stride=1,
        #         padding=3,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(
        #         in_channels=500,
        #         out_channels=out_dim,
        #         kernel_size=5,
        #         stride=1,
        #         padding=2,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(
        #         in_channels=500,
        #         out_channels=out_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        # )
        # self.tdnn = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=num_features,
        #         out_channels=500,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(num_features=500, affine=False),
        #     nn.Conv1d(
        #         in_channels=500,
        #         out_channels=500,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(num_features=500, affine=False),
        #     nn.Conv1d(
        #         in_channels=500,
        #         out_channels=out_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(num_features=out_dim, affine=False),
        # )
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=500,
                kernel_size=11,
                stride=1,
                padding=5,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=500,
                out_channels=500,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(num_features=500, affine=False),
            nn.Conv1d(
                in_channels=500,
                out_channels=out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(num_features=out_dim, affine=False),
        )

    def forward(self, src):
        x = src.permute(0, 2, 1)
        x = self.tdnn(x)
        x = x.permute(0, 2, 1)
        return x


class SpeechEncoderPrenet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(self, args, embed_tokens, vocab, mask_whole_word, mask_idx, 
                 frame_target_classes,
                 sampling_nums= 1,
                 policy_feat_dim=768,
                 use_transformer_policy=True,
                 use_softmax_soft_pool=False,
                 mask_length = "span-poisson", poisson_lambda = 3.5,
                 mask = 0.3, mask_random = 0.1, iid_noise_target = False, replace_length=1, uni_mask_idxs = None, word_freq=12):
        super(SpeechEncoderPrenet, self).__init__()
        new_args = copy.deepcopy(args)
        
        self.sampling_nums = sampling_nums
        self.policy_feat_dim = policy_feat_dim
        

        self.use_transformer_policy = use_transformer_policy
        if not use_transformer_policy:
            self.policy_network = TdnnSegmenter(policy_feat_dim, 500)
            self.policy_network_logits = nn.Linear(500, 1)
            self.frame_target_network_logits = nn.Linear(500, frame_target_classes)
        else:
            pos_enc_class = (
            ScaledPositionalEncoding if args.enc_use_scaled_pos_enc else PositionalEncoding
            )
            self.policy_pre_proj = None
            if policy_feat_dim != new_args.encoder_embed_dim:
                self.policy_pre_proj = nn.Linear(policy_feat_dim, new_args.encoder_embed_dim)
                
            self.frame_pos_enc = pos_enc_class(new_args.encoder_embed_dim, 
                                               args.transformer_enc_positional_dropout_rate, 
                                               max_len=int(args.max_text_positions * word_freq))
            self.policy_network = TransformerEncoder(new_args, tgt_dict=None, embed_tokens=None, src_dict=None, src_embed_tokens=None)
            self.policy_network_logits = nn.Linear(new_args.encoder_embed_dim, 1)
            self.frame_target_network_logits = nn.Linear(new_args.encoder_embed_dim, frame_target_classes)
        # self.frame_encoder_network = TransformerEncoder(new_args, tgt_dict=None, embed_tokens=None, src_dict=None, src_embed_tokens=None)
        # self.addtional_encoder_network = TransformerEncoder(new_args, tgt_dict=None, embed_tokens=None, src_dict=None, src_embed_tokens=None)
        # self.policy_network_logits = nn.Linear(768, 1)
        self.padding_idx = embed_tokens.padding_idx
        # define encoder prenet
        # get positional encoding class
        
        self.vocab = vocab
        # print('nspecial: ', vocab.nspecial)
        self.mask_whole_word = mask_whole_word
        self.mask_idx = mask_idx
        self.quantizer = KmeansVectorQuantizer(dim=new_args.input_feat_per_channel, nspecial=self.vocab.nspecial, num_vars=args.km_size, init_clus_dir=args.init_clus_dir)
        pos_enc_class = (
            ScaledPositionalEncoding if args.enc_use_scaled_pos_enc else PositionalEncoding
        )
        
        self.random_ratio = mask_random
        self.mask_ratio = mask
        self.mask_span_distribution = None
        if mask_length == "span-poisson":
            _lambda = poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)
            
        self.iid_noise_target = iid_noise_target
        self.uni_mask_idxs = uni_mask_idxs
        self.encoder_prenet = nn.Sequential(embed_tokens,
                                            pos_enc_class(args.encoder_embed_dim, 
                                                          args.transformer_enc_positional_dropout_rate, 
                                                          max_len=args.max_text_positions)
                                            
        )
        self.replace_length = replace_length
        self.word_freq = int(word_freq)
        self.use_softmax_soft_pool = use_softmax_soft_pool
    
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start
    
    def add_whole_word_mask(self, source, p):
        source_ori = source.clone()
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return source, source_ori, None
            # if num_to_mask == 0:
            #     return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            ).to(source.device)

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source.device)
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source.device)

                assert source_length - 1 not in indices

        if not self.iid_noise_target:
            unmasked_source = source_ori[to_keep]
            source = source[to_keep]
            target = None
            assert len(unmasked_source) == len(source)
        else:
            print('Warning! iid_noise_target is not False')
            ## Prepare source
            source_mask_idx = (source == self.mask_idx).nonzero().view(-1)
            source[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]
            source = source[to_keep]

            ## Prepare target
            to_keep[source_mask_idx] = 0

            # source_mask_idx: from [a, b, c, ...] to [a, b + 1, c + 2, ...]
            source_mask_idx = source_mask_idx + torch.arange(source_mask_idx.size(0))
            # target: source_length + mask_length
            target = source_ori.new_zeros(source_mask_idx.size(0) + source_ori.size(0))
            # target: [0, 0, 0, X, 0, 0, Y, ....]
            target[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]

            target_to_keep = to_keep.new_zeros(source_mask_idx.size(0) + source_ori.size(0))

            # Copy original value to target and target_to_keep
            target_to_keep[target == 0] = to_keep
            target_to_keep[-1] = 0
            target[target == 0] = source_ori

            target = target[~target_to_keep]

        # if num_inserts > 0:
        #     source, unmasked_source = self.add_insertion_noise(source, unmasked_source, num_inserts / source.size(0))

        return source, unmasked_source, target
    
    def add_insertion_noise(self, tokens, unmasked_tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1).to(tokens.device)
        unmasked_result = torch.LongTensor(n + len(tokens)).fill_(-1).to(tokens.device)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        ).to(tokens.device)
        unmasked_result[noise_indices[num_random:]] = result[noise_indices[num_random:]].clone()
        unmasked_result[noise_indices[:num_random]] = result[noise_indices[:num_random]].clone()
        

        result[~noise_mask] = tokens
        unmasked_result[~noise_mask] = unmasked_tokens

        assert (result >= 0).all()
        return result, unmasked_result
    
    
    def add_whole_word_mask_soft_source(self, source_softmax, p):
        source = source_softmax.argmax(dim = -1)
        # print('0', source_softmax.argmax(-1))
        source_ori = source_softmax.clone()
        # print(source_softmax.size())
        source_vocab_dim = source_softmax.size(-1)
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source_softmax, source_ori, None

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            # if num_to_mask == 0:
            #     return source_softmax, source_ori, None
            # if num_to_mask == 0:
            #     return self.add_insertion_noise_soft_source(source_softmax, source_ori, num_inserts / source.size(0))
            if num_to_mask == 0:
                return source_softmax, source_ori, None
            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            mask_tensor = source_softmax.new_zeros(len(indices), source_vocab_dim)
            mask_tensor[:, self.mask_idx] = 1
            source_softmax[indices] = source_softmax[indices] - source_softmax[indices].detach() + mask_tensor

            mask_random_tensor = torch.randint(
                4, len(self.vocab) - 1, size=(mask_random.sum(),)
            ).to(source_softmax.device)
            
            mask_random_tensor = F.one_hot(mask_random_tensor, num_classes = source_vocab_dim)
            
            source_softmax[indices[mask_random]] = source_softmax[indices[mask_random]] - source_softmax[indices[mask_random]].detach() + mask_random_tensor
            # print('1', source_softmax.argmax(-1))
        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    mask_tensor = source_softmax.new_zeros(len(indices), source_vocab_dim)
                    mask_tensor[:, self.mask_idx] = 1
                    source_softmax[indices] = source_softmax[indices] - source_softmax[indices].detach() + mask_tensor

                    mask_random_tensor = torch.randint(
                        4, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source_softmax.device)
                    
                    mask_random_tensor = F.one_hot(mask_random_tensor, num_classes = source_vocab_dim)
                    
                    source_softmax[indices[mask_random]] = source_softmax[indices[mask_random]] - source_softmax[indices[mask_random]].detach()+ mask_random_tensor

        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    mask_tensor = source_softmax.new_zeros(len(indices), source_vocab_dim)
                    mask_tensor[:,  self.mask_idx] = 1
                    source_softmax[indices] = source_softmax[indices] - source_softmax[indices].detach() + mask_tensor

                    mask_random_tensor = torch.randint(
                        4, len(self.vocab), size=(mask_random.sum(),)
                    ).to(source_softmax.device)
                    
                    mask_random_tensor = F.one_hot(mask_random_tensor, num_classes = source_vocab_dim)
                    
                    source_softmax[indices[mask_random]] = source_softmax[indices[mask_random]] - source_softmax[indices[mask_random]].detach()+ mask_random_tensor


                assert source_length - 1 not in indices

        if not self.iid_noise_target:
            unmasked_source_ret = source_ori[to_keep]
            source_softmax_ret = source_softmax[to_keep]
            target = None
            assert len(unmasked_source_ret) == len(source_softmax_ret)
        else:
            print('Warning! iid_noise_target is not False')
            # ## Prepare source
            # source_mask_idx = (source == self.mask_idx).nonzero().view(-1)
            # source[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]
            # source = source[to_keep]

            ## Prepare source
            source_mask_idx = (source == self.mask_idx).nonzero().view(-1)
            
            uni_mask_tensor = self.uni_mask_idxs[:source_mask_idx.size(0)]
            uni_mask_tensor = F.one_hot(uni_mask_tensor, num_classes = source_vocab_dim)
            source_softmax[source_mask_idx] = source_softmax[source_mask_idx] - source_softmax[source_mask_idx].detach() + uni_mask_tensor * 1
            source_softmax_ret = source_softmax[to_keep]
            unmasked_source_ret = source_ori[to_keep]

            ## Prepare target
            to_keep[source_mask_idx] = 0

            # source_mask_idx: from [a, b, c, ...] to [a, b + 1, c + 2, ...]
            source_mask_idx = source_mask_idx + torch.arange(source_mask_idx.size(0))
            # target: source_length + mask_length
            target = source.new_zeros(source_mask_idx.size(0) + source_ori.size(0))
            # target: [0, 0, 0, X, 0, 0, Y, ....]
            target[source_mask_idx] = self.uni_mask_idxs[:source_mask_idx.size(0)]

            target_to_keep = to_keep.new_zeros(source_mask_idx.size(0) + source_ori.size(0))

            # Copy original value to target and target_to_keep
            target_to_keep[target == 0] = to_keep
            target_to_keep[-1] = 0
            target[target == 0] = source_ori

            target = target[~target_to_keep]

        # # print('4', source_softmax_ret.argmax(-1))
        # if num_inserts > 0:
        #     source_softmax_ret, unmasked_source_ret, _ = self.add_insertion_noise_soft_source(source_softmax_ret, unmasked_source_ret, num_inserts / source.size(0))

        return source_softmax_ret, unmasked_source_ret, target
    
    def add_insertion_noise_soft_source(self, tokens, unmasked_tokens, p):
        if p == 0.0:
            return tokens, unmasked_tokens, None

        # print(tokens.size(), unmasked_tokens.size())

        source_vocab_dim = tokens.size(-1)

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.FloatTensor(n + len(tokens), source_vocab_dim).fill_(0).to(tokens.device)
        unmasked_result = torch.FloatTensor(n + len(tokens), source_vocab_dim).fill_(0).to(tokens.device)

        num_random = int(math.ceil(n * self.random_ratio))

        mask_tensor = tokens.new_zeros(len(noise_indices[num_random:]), source_vocab_dim)
        mask_tensor[:, self.mask_idx] = 1

        mask_random_tensor = torch.randint(
            4, len(self.vocab), size=(num_random,)
        ).to(tokens.device)
        
        mask_random_tensor = F.one_hot(mask_random_tensor, num_classes = source_vocab_dim).float()

        result[noise_indices[num_random:]] = mask_tensor
        result[noise_indices[:num_random]] = mask_random_tensor
        unmasked_result[noise_indices[num_random:]] = result[noise_indices[num_random:]].clone()
        unmasked_result[noise_indices[:num_random]] = result[noise_indices[:num_random]].clone()
        
        result[~noise_mask] = result[~noise_mask] * 0 + 1 * tokens
        unmasked_result[~noise_mask] =  unmasked_result[~noise_mask] * 0 + 1 * unmasked_tokens

        # assert (result >= 0).all()
        return result, unmasked_result, None
        
    def collater_audio(self, features):
        sizes = [len(s) for s in features]
        # print('audio', sizes)
        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        for i, (f, size) in enumerate(zip(features, sizes)):
            real_size = size
            collated_features[i, :real_size] = f[:real_size]
            padding_mask[i, size:] = True

        return collated_features, padding_mask, target_size

    def get_cur_pooled_src_sample(self, curr_feats, curr_length, curr_boundaries):
        
        boundary_cumsum = torch.cumsum(curr_boundaries, dim = 0).unsqueeze(-1)
        num_segs = round(boundary_cumsum.squeeze()[-1].item())
        # print(curr_length, num_segs)
        incremental_arr = torch.ones(curr_length, num_segs).to(curr_feats.device).cumsum(dim = -1)
        # print(num_segs, incremental_arr.size())
        tanh = nn.Tanh()
        # print(incremental_arr.size(), boundary_cumsum.size())
        mean_matrix = 1 - tanh(10 * torch.abs(incremental_arr - boundary_cumsum))
        mean_matrix = mean_matrix / (1e-8 + mean_matrix.sum(dim=0, keepdim = True))
        # print(mean_matrix, curr_boundaries)
        curr_pooled_src = torch.matmul(mean_matrix.transpose(0,1), curr_feats)
        return curr_pooled_src

    # def get_cur_pooled_src_sample(self, curr_feats, curr_length, curr_boundaries):
    #     boundary_locs = torch.where(curr_boundaries == 1)[0]
    #     boundary_starts = boundary_locs[:-1]
    #     boundary_ends = boundary_locs[1:]

    #     curr_pooled_src = []

    #     for idx, (start, end) in enumerate(zip(boundary_starts, boundary_ends)):
    #         curr_pooled_src.append(curr_feats[start:end].mean(dim = -2))

    #     curr_pooled_src = torch.stack(curr_pooled_src, dim = 0)
    #     return curr_pooled_src
    
    def get_cur_pooled_src_sample_logit_segment(self, curr_feats, curr_length, curr_boundaries):
        preds = (curr_boundaries > 0.5).long().cumsum(-1)
        # print(preds)
        
        logits = curr_feats

        tsz, csz = logits.shape
    
        u, idx, c = preds.cpu().unique_consecutive(return_inverse=True, return_counts=True)
        
        # compute soft counts 
        sc = curr_boundaries.cumsum(-1)
        
        # construct pooling matrix of size (N_seg, T)
        ns = u.numel()
        w = torch.arange(ns).repeat(tsz, 1).t()
        w = w.to(curr_boundaries.device)
        # print(w.size(), sc.size())
        w = - torch.abs(w - sc) * 10.
        w = torch.softmax(w, dim=-1)
        
        # perform pooling on logits
        new_logits = torch.mm(w, logits) 

        return new_logits

    def forward(self, src, padding_mask, boundaries, discrete_frame_cluster = None, discrete_batch = None, audio_tokens = None, use_boundaries= False, debugging = False, task_name='speech_text_pretrain'):

        if self.use_transformer_policy:
            if self.policy_pre_proj is not None:
                pre_pos_enc_src = self.policy_pre_proj(src)
            else:
                pre_pos_enc_src = src
            pos_enc_src = self.frame_pos_enc(pre_pos_enc_src)
            policy_encoder_output = self.policy_network(pos_enc_src, padding_mask, tgt_layer=None)
            policy_encoder_states = policy_encoder_output["encoder_out"][0].transpose(0, 1)
            policy_encoder_padding_mask = policy_encoder_output["encoder_padding_mask"][0]

        else:
            policy_encoder_states = self.policy_network(src)
            policy_encoder_padding_mask = padding_mask
            
        policy_logits = self.policy_network_logits(policy_encoder_states)
        # policy_logits[policy_encoder_padding_mask, :] = float("-inf")
        frame_target_logits = self.frame_target_network_logits(policy_encoder_states)
        # frame_target_logits[policy_encoder_padding_mask, :] = float("-inf")

        # policy_probs = nn.Sigmoid()(policy_logits)
        sigmoid = nn.Sigmoid()
        b_soft = sigmoid(policy_logits)
        b_hard = sigmoid(1000 * policy_logits)
        b_diff = b_hard - b_soft
        hard_policy_probs = b_soft + b_diff.detach()
        soft_policy_probs = b_soft
    

        # db = copy.deepcopy(discrete_batch)
        # discrete_batch = None
        # audio_tokens = None
        # use_boundaries = False
        if use_boundaries and discrete_batch is not None:
            batch = discrete_batch
            #batch["prenet_out"] = self.addtional_encoder_network(self.encoder_prenet(batch["net_input"]["src_tokens"]), batch["net_input"]["src_tokens"].eq(self.padding_idx))["encoder_out"][0].transpose(0, 1)
            batch["prenet_out"] = self.encoder_prenet(batch["net_input"]["src_tokens"])
            batch["padding_mask"] = batch["net_input"]["src_tokens"].eq(self.padding_idx)
            batch["target_padding_mask"] = batch["target"].eq(self.padding_idx)
        
            
            
            return [batch], policy_logits, frame_target_logits, policy_encoder_padding_mask
            
            
        elif use_boundaries and audio_tokens is not None:
            # print('audio tokens is not none')
            batch = fake_collate(audio_tokens)
            #batch["prenet_out"] = self.addtional_encoder_network(self.encoder_prenet(batch["net_input"]["src_tokens"]), batch["net_input"]["src_tokens"].eq(self.padding_idx))["encoder_out"][0].transpose(0, 1)
            batch["prenet_out"] = self.encoder_prenet(batch["net_input"]["src_tokens"])
            batch["padding_mask"] = batch["net_input"]["src_tokens"].eq(self.padding_idx)
            batch["target_padding_mask"] = batch["padding_mask"]
            
            
            return [batch], policy_logits, frame_target_logits, policy_encoder_padding_mask
        
        elif use_boundaries:
            pooled_src = []
            pooled_unmasked_src = []
            pooled_target = []
            pooled_discrete_lprobs = []
            bsz = src.size(0)
            length_check = (~policy_encoder_padding_mask).sum(dim=-1)
            for b_idx in range(bsz):
                curr_length = length_check[b_idx]
                curr_boundaries = boundaries[b_idx][:curr_length]
                curr_feats = src[b_idx][:curr_length, :]
                curr_pooled_src = self.get_cur_pooled_src_sample(curr_feats, curr_length, curr_boundaries)

                discrete_idx, discrete_lprobs, _ = self.quantizer(curr_pooled_src, sampling = False)
                if "speech_text_pretrain" in task_name:
                    discrete_idx = torch.cat([torch.LongTensor([self.vocab.bos()]).to(discrete_idx.device), 
                                            discrete_idx,
                                            torch.LongTensor([self.vocab.eos()]).to(discrete_idx.device)])

                source, target = discrete_idx, discrete_idx.detach().clone()
                if self.mask_ratio > 0 and "speech_text_pretrain" in task_name:
                    source, unmasked_source, new_target = self.add_whole_word_mask(source, self.mask_ratio)
                    assert len(source) == len(unmasked_source)
                    if new_target is not None:
                        target = new_target
                    # target = target_ori[len(target)//2:]
                    # source = source_ori[:len(source)//2,...]
                    # unmasked_source = source.clone()
                    # pooled_src.append(source)
                    # pooled_unmasked_src.append(unmasked_source)
                    # pooled_target.append(target)
                    # pooled_discrete_lprobs.append(discrete_lprobs)
                else:
                    pooled_src.append(source)
                    pooled_target.append(target)
                    pooled_discrete_lprobs.append(discrete_lprobs)


                    
            batch = collate(pooled_src, pooled_unmasked_src, pooled_target, self.padding_idx, self.vocab.eos())
            #batch["prenet_out"] = self.addtional_encoder_network(self.encoder_prenet(batch["net_input"]["src_tokens"]), batch["net_input"]["src_tokens"].eq(self.padding_idx))["encoder_out"][0].transpose(0, 1)
            batch["prenet_out"] = self.encoder_prenet(batch["net_input"]["src_tokens"])
            batch["padding_mask"] = batch["net_input"]["src_tokens"].eq(self.padding_idx)
            batch["target_padding_mask"] = batch["target"].eq(self.padding_idx)
            
            return [batch],  policy_logits, frame_target_logits, policy_encoder_padding_mask
                
        else:
            length_check = (~policy_encoder_padding_mask).sum(dim=-1)
            batches = []
            bsz = src.size(0)
            
            hard_policies = []
            if self.use_softmax_soft_pool:
                hard_policies.append(soft_policy_probs)
            else:
                hard_policies.append(hard_policy_probs)
                
            ret_policies = hard_policies

            # print(ret_policy.size(), boundaries.size())
            for sampling_idx in range(len(hard_policies)):
                ret_policy = ret_policies[sampling_idx]
                hard_policy = hard_policies[sampling_idx]
    
                
                pooled_src = []
                pooled_unmasked_src = []
                pooled_target = []
                pooled_discrete_lprobs = []
                pooled_code_loss = []
                
                batch_sum_words_probs = []
                batch_sum_nonwords_probs = []
                batch_sum_consecutive_probs = []
                
                for b_idx in range(bsz):
                
                
                    curr_length = length_check[b_idx]
                    curr_boundaries = boundaries[b_idx][:curr_length]
                    # print('boundary_sum:', torch.sum(curr_boundaries))
                    # print('predicted boundary_sum', torch.sum(hard_policy_probs[b_idx][:curr_length]))
                    # curr_boundaries_hard = curr_boundaries.new_zeros(curr_boundaries.size())
                    # curr_boundaries_hard[curr_boundaries > 0.5] = 1
                    # # curr_boundaries = curr_boundaries_hard
                    # curr_boundaries = boundaries[b_idx][:curr_length]
                    # print(curr_boundaries)
                    # print(boundaries[b_idx][:curr_length])
                    # print('-----------------------------------------------------------')
                    curr_feats = src[b_idx][:curr_length, :]
                    
                    sum_words_probs = torch.sum(hard_policy_probs[b_idx][:curr_length][hard_policy_probs[b_idx][:curr_length] > 0.5,...])
                    sum_nonwords_probs = torch.sum(-hard_policy_probs[b_idx][:curr_length][hard_policy_probs[b_idx][:curr_length] < 0.5,...] + 1)
                    
                    chunks = int(curr_length // self.word_freq)
                    trimmed_length = int(chunks * self.word_freq)
                    trimmed_policy = hard_policy_probs[b_idx][:trimmed_length].squeeze().reshape(chunks, self.word_freq)
                    trimmed_policy_ones = trimmed_policy * (trimmed_policy > 0.5)
                    trimmed_policy_zeros = (1-trimmed_policy) * (trimmed_policy <= 0.5)

                    sum_consecutive_probs = F.l1_loss(trimmed_policy_ones.sum(dim = -1), trimmed_policy_ones.new_ones(chunks), reduction="sum") + F.l1_loss(trimmed_policy_zeros.sum(dim = -1), -trimmed_policy_ones.new_ones(chunks) + self.word_freq, reduction="sum")
                    batch_sum_words_probs.append(sum_words_probs)
                    batch_sum_nonwords_probs.append(sum_nonwords_probs)
                    batch_sum_consecutive_probs.append(sum_consecutive_probs)
                    
                    if len(torch.where(hard_policy[b_idx].squeeze()[:curr_length] > 0.5)[0]) > 2:
                        curr_boundaries = hard_policy[b_idx].squeeze()[:curr_length]
                    else:
                        # ret_policy[b_idx, :, 0] = 0
                        # boundary_locs = torch.where(boundaries[b_idx] == 1)[0]
                        # ret_policy[b_idx, boundary_locs, 0] = 1

                        logger.info(f'WARNING! BOUNDARY SEGMENTER FAILED COMPLETELY ON AN UTTERANCE!')
                        
                        
                    if self.use_softmax_soft_pool:
                        curr_pooled_src = self.get_cur_pooled_src_sample_logit_segment(curr_feats, curr_length, curr_boundaries)
                    else:
                        # print('correct')
                        curr_pooled_src = self.get_cur_pooled_src_sample(curr_feats, curr_length, curr_boundaries)
                        
                    

                    discrete_idx, discrete_lprobs, code_loss = self.quantizer(curr_pooled_src, sampling=True)
                    
                    pooled_code_loss.append(code_loss)
                    # print('-1', discrete_idx.argmax(-1))
                    # if task_name != 's2t':
                    #     bos_tensor = torch.zeros(1, discrete_idx.size(-1)).to(discrete_idx.device).long()
                    #     bos_tensor[0, self.vocab.bos()] = 1
                        
                    #     eos_tensor = torch.zeros(1, discrete_idx.size(-1)).to(discrete_idx.device).long()
                    #     eos_tensor[0, self.vocab.eos()] = 1
                    #     discrete_idx= torch.cat([bos_tensor, 
                    #                             discrete_idx,
                    #                             eos_tensor], dim = 0)
                    #     # print('-1', discrete_idx.argmax(-1))
                    # source_no_add, target = discrete_idx, discrete_idx.detach().clone().argmax(-1)
                    # target = target + 4
                    # source = source_no_add.new_zeros(source_no_add.size(0), source_no_add.size(1) + 5)
                    # source[:, 4:-1] = source_no_add
                    
                    source_no_add, target = discrete_idx, discrete_idx.detach().clone().argmax(-1)
                    target = target + 4
                    source = source_no_add.new_zeros(source_no_add.size(0), source_no_add.size(1) + 5)
                    source[:, 4:-1] = source_no_add
                    # print('pre_mask:', source.size(), target.size())
                    if "speech_text_pretrain" in task_name:
                        bos_tensor = torch.zeros(1, source.size(-1)).to(discrete_idx.device).long()
                        bos_tensor[0, self.vocab.bos()] = 1
                        
                        eos_tensor = torch.zeros(1, source.size(-1)).to(discrete_idx.device).long()
                        eos_tensor[0, self.vocab.eos()] = 1
                        source = torch.cat([bos_tensor, 
                                                source,
                                                eos_tensor], dim = 0)
                        bos_idx = (torch.ones( 1).to(discrete_idx.device) * self.vocab.bos()).long()
                        eos_idx = (torch.ones( 1).to(discrete_idx.device) * self.vocab.eos()).long()
                        target = torch.cat([bos_idx, target, eos_idx], dim = 0)
                    # print(source.argmax(-1))
                    if self.mask_ratio > 0 and "speech_text_pretrain" in task_name:
                        source, unmasked_source, new_target = self.add_whole_word_mask_soft_source(source, self.mask_ratio)
                        assert len(source) == len(unmasked_source)
                        if new_target is not None:
                            print('WARNING: new target is not None!')
                            target = new_target
                        # print('1', source.argmax(-1))
                        # pooled_src.append(source)
                        # pooled_unmasked_src.append(unmasked_source)
                        # pooled_target.append(target)
                        # pooled_discrete_lprobs.append(discrete_lprobs)
                        # target = target[len(target)//2:]
                        # source = source[:len(source)//2,...]
                        # unmasked_source = source.clone()
                        pooled_src.append(source)
                        pooled_unmasked_src.append(unmasked_source)
                        pooled_target.append(target)
                        pooled_discrete_lprobs.append(discrete_lprobs)
                        
                    else:
                        pooled_src.append(source)
                        pooled_target.append(target)
                        pooled_discrete_lprobs.append(discrete_lprobs)

                    # print('post_mask:', source.size(), target.size())
            
                batch = collate_onehot(pooled_src, pooled_unmasked_src, pooled_target, self.padding_idx)
                batch["prenet_out"] = torch.matmul(batch["net_input"]["src_tokens"], self.encoder_prenet[0].weight)
                # print(batch["prenet_out"], torch.isinf(batch["prenet_out"]).sum())
                #batch["prenet_out"] =  self.addtional_encoder_network(self.encoder_prenet[1](batch["prenet_out"]), batch["net_input"]["src_padding_mask"])["encoder_out"][0].transpose(0, 1)
                batch["prenet_out"] =  self.encoder_prenet[1](batch["prenet_out"])
                batch["target_padding_mask"] = batch["target"].eq(self.padding_idx)
                batch["hard_policy"] = ret_policy
                batch_sum_words_probs = [b.reshape(1) for b in batch_sum_words_probs]
                batch_sum_nonwords_probs = [b.reshape(1) for b in batch_sum_nonwords_probs]
                # print(batch_sum_words_probs)
                # print(batch_sum_nonwords_probs)
                batch["sum_after_mean_word_probs"] = torch.cat(batch_sum_words_probs)
                batch["sum_after_mean_nonword_probs"] = torch.cat(batch_sum_nonwords_probs)
                batch["sum_after_mean_consecutive_probs"] = sum(batch_sum_consecutive_probs)
                batch["padding_mask"] = batch["net_input"]["src_padding_mask"]
                batch["code_loss"] = sum(pooled_code_loss) 
                # print(batch["net_input"]["src_tokens"][0].argmax(dim=-1), db["net_input"]["src_tokens"][0])
        
                
                batches.append(batch)
                
                
        return batches, policy_logits, frame_target_logits, policy_encoder_padding_mask
    
    
    def forward_tokens(self, src_tokens, weight_mm= False):
        if not weight_mm:
            prenet_out = self.encoder_prenet(src_tokens)
        else:
            prenet_out = torch.matmul(src_tokens, self.encoder_prenet[0].weight)
            prenet_out = self.encoder_prenet[1](prenet_out)
            
        return prenet_out
    
    # def get_padding_mask(self, src_tokens):
    #     return src_tokens.eq(self.padding_idx)