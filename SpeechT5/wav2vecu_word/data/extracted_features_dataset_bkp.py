# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


class ExtractedFeaturesDataset(FairseqDataset):
    def __init__(
        self,
        audio_path,
        image_path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        label_dict=None,
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.label_dict = label_dict

        if labels is not None:
            assert label_dict is not None

        self.a_sizes = []
        self.v_sizes = []
        self.a_offsets = []
        self.v_offsets = []
        self.labels = []
        self.aux_tgt = None

        audio_path = os.path.join(audio_path, split)
        image_path = os.path.join(image_path, split)
        self.audio_data = np.load(audio_path + ".npy", mmap_mode="r")
        # TODO Insert silences between image features
        self.image_data = np.load(image_path + ".npy", mmap_mode="r")

        a_offset = 0
        v_offset = 0
        skipped = 0

        if not os.path.exists(path + f".{labels}"):
            labels = None

        with open(audio_path + ".lengths", "r") as a_len_f, open(
            image_path + ".lengths", "r") as v_len_f, open(
            path + f".{labels}", "r"
        ) if labels is not None else contextlib.ExitStack() as lbl_f:
            for a_line, v_line in zip(a_len_f, v_line):
                a_length = int(line.rstrip())
                v_length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).rstrip().split()
                if a_length >= min_length and (
                    max_length is None or a_length <= max_length
                ):
                    self.a_sizes.append(a_length)
                    self.a_offsets.append(a_offset)
                    self.v_sizes.append(v_length)
                    self.v_offsets.append(v_offset)
                    if lbl is not None:
                        self.labels.append(lbl)
                a_offset += a_length
                v_offset += v_length

        self.a_sizes = np.asarray(self.a_sizes)
        self.a_offsets = np.asarray(self.a_offsets)
        self.v_sizes = np.asarray(self.v_sizes)
        self.a_offsets = np.asarray(self.a_offsets)
        
        if aux_target_postfix is not None:
            if not os.path.exists(path+f".{aux_target_postfix}"):
                logger.info(f"auxaliry target for {split} missing")
            else:
                with open(path+f".{aux_target_postfix}", "r") as t_f:
                    self.aux_tgt = [
                        torch.LongTensor(list(map(int,seg.strip().split())))\
                                    for seg in t_f]
 
        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        a_offset = self.a_offsets[index]
        a_end = self.a_sizes[index] + a_offset
        v_offset = self.v_offsets[index]
        v_end = self.v_sizes[index] + v_offset

        afeats = torch.from_numpy(self.audio_data[a_offset:a_end].copy()).float()
        vfeats = torch.from_numpy(self.image_data[v_offset:v_end].copy()).float()

        res = {"id": index, "audio_features": afeats, "image_features": vfeats}
        if len(self.labels) > 0:
            res["target"] = self.label_dict.encode_line(
                self.labels[index],
                line_tokenizer=lambda x: x,
                append_eos=False,
            )
        
        if self.aux_tgt:
            res["aux_target"] = self.aux_tgt[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        audio_features = [s["audio_features"] for s in samples]
        a_sizes = [len(s) for s in audio_features]

        image_features = [s["image_features"] for s in samples]
        v_sizes = [len(s) for s in image_features]

        a_target_size = max(a_sizes)
        v_target_size = max(v_sizes)

        collated_audio_features = audio_features[0].new_zeros(
            len(audio_features), a_target_size, audio_features[0].size(-1)
        )
        collated_image_features = image_features[0].new_zeros(
            len(image_features), v_target_size, image_features[0].size(-1)
        )

        a_padding_mask = torch.BoolTensor(collated_audio_features.shape[:-1]).fill_(False)
        v_padding_mask = torch.BoolTensor(collated_image_features.shape[:-1]).fill_(False)
        for i, (f, size) in enumerate(zip(audio_features, a_sizes)):
            collated_audio_features[i, :size] = f
            a_padding_mask[i, size:] = True

        for i, (f, size) in enumerate(zip(image_features, v_sizes)):
            collated_image_features[i, :size] = f
            v_padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "audio_features": collated_audio_features,
                "image_features": collated_image_features,
                "audio_padding_mask": a_padding_mask,
                "image_padding_mask": v_padding_mask,
            },
        }

        if len(self.labels) > 0:
            target = data_utils.collate_tokens(
                [s["target"] for s in samples],
                pad_idx=self.label_dict.pad(),
                left_pad=False,
            )
            res["target"] = target
        
        if self.aux_tgt:
            idxs = torch.nn.utils.rnn.pad_sequence(
                [s["aux_target"] for s in samples],
                batch_first=True,
                padding_value=-1,
            )
            res["net_input"]["aux_target"] = idxs
        
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]
