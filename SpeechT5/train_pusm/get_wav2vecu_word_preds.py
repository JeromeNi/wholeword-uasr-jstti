import logging
import os.path as op
from argparse import Namespace
from collections import OrderedDict
import pickle

from matplotlib.pyplot import text

import torch
from fairseq.data import (
    Dictionary, 
    encoders, 
    PrependTokenDataset,
    AppendTokenDataset, 
    data_utils, 
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
import fairseq
from easydict import EasyDict as edict
import os
import pickle
import numpy as np
import json
from fairseq.tasks.hubert_pretraining import LabelEncoder
from fairseq.data import encoders
import editdistance 


def add_args(parser):
    parser.add_argument("data", help="manifest root path")
    
    parser.add_argument(
        "--split",
        default="speech_train|text_train",
        type=str,
        help="split to use",
    )

    parser.add_argument(
        "--ckpt-path",
        default="",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument(
        "--user-dir",
        default="",
        type=str,
        help="user dir path",
    )   
        

def calc_uer(hf, rf):

    errs = 0
    count = 0
    # h and r are lists
    for h, r in zip(hf, rf):
        errs += editdistance.eval(r.strip().split(), h.strip().split())
        count += len(r.strip().split())

    return errs / count


from itertools import groupby
def calc_uer_remove_rep(hf, rf):

    errs = 0
    count = 0
    # h and r are lists
    for h, r in zip(hf, rf):
        errs += editdistance.eval(r.strip().split(), [key for key, _group in groupby(h.strip().split())])
        count += len(r.strip().split())

    return errs / count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    
    bad_files = []
    bad_idxs = []
    if 'train' == args.split:
        with open(os.path.join(args.data, 'train_bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()
    elif 'valid_true' == args.split:
        with open(os.path.join(args.data, 'valid_true_bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()
    else:
        with open(os.path.join(args.data, 'valid_bnd_errors.txt'), 'r') as f:
            bad_lines = f.readlines()            

    for bl in bad_lines:
        bad_file = bl.strip().split(maxsplit = 1)[0]
        bad_files.append(bad_file)

    bad_files = set(bad_files)

    filenames = []
    with open(os.path.join(args.data, f"{args.split}.files"), 'r') as f:
        file_lines = f.readlines()
    for idx, file_line in enumerate(file_lines):
        cur_filename = file_line.strip()
        if cur_filename not in bad_files:
            filenames.append(cur_filename)
        else:
            bad_idxs.append(idx)
    
    
    module_options = edict({'user_dir': args.user_dir})
    utils.import_user_module(module_options)
    
    (model, cfg, task) = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.ckpt_path])
    model = model[0].eval().to("cuda:0")        
    
    data = np.load(os.path.join(args.data, f"{args.split}.npy"), mmap_mode="r")
    
    text_dict = Dictionary.load(op.join(args.data,".." ,"dict.txt"))
    
    sizes = []
    offsets = []
    
    offset = 0

    with open(os.path.join(args.data, f"{args.split}.lengths"), "r") as len_f:
        for file_idx, line in enumerate(len_f):
            length = int(line.rstrip())
            if file_idx not in bad_idxs:
                sizes.append(length)
                offsets.append(offset)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    
    labels = []
    with open(os.path.join(args.data, f"{args.split}.wrd"), "r") as lbl_f:
        for file_idx, line in enumerate(lbl_f):
            transcript = line.rstrip()
            if file_idx not in bad_idxs:
                labels.append(transcript)

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)
    
    
    print(len(filenames), len(sizes))
    all_preds = []
    # with open(os.path.join(os.path.dirname(args.ckpt_path), f"{args.split}_preds.txt"), 'w') as fw:
    for index in range(len(sizes)):
        offset = offsets[index]
        end = sizes[index] + offset
        filename = filenames[index]
        feats = torch.from_numpy(data[offset:end].copy()).float()
        padding_mask = torch.BoolTensor(feats.size()).fill_(False)
        
        if len(feats.shape) == 1:
            feats = feats.unsqueeze(-1)
            
        feats = feats.unsqueeze(0).to("cuda:0")
        padding_mask = padding_mask.unsqueeze(0).to("cuda:0")
        
        results = model.generator(dense_x = feats, tokens = None, dense_padding_mask = padding_mask)
        
        z = results["dense_x"].argmax(-1).squeeze(0)
        pred_str = text_dict.string(z)
        all_preds.append(pred_str)
            # fw.write(filename + '\t' + pred_str + '\n')
            
    uer_1 = calc_uer_remove_rep(all_preds, labels)
    uer_2 = calc_uer(all_preds, labels)
    print('WER:', uer_1, uer_2)

    if uer_1 > uer_2:
        with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.split}_preds.txt'), 'w') as f:
            for cur_output_pred, cur_output_fn in zip(all_preds, filenames):
                f.write(cur_output_fn + '\t' + cur_output_pred + '\n')
    else:
        with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.split}_preds.txt'), 'w') as f:
            for cur_output_pred, cur_output_fn in zip(all_preds, filenames):
                f.write(cur_output_fn + '\t' + " ".join([key for key, _group in groupby(cur_output_pred.strip().split())]) + '\n')
            
    with open(op.join(os.path.dirname(args.ckpt_path), f'predictions_with_fn', f'{args.split}_ref.txt'), 'w') as f:
        for cur_output_gt, cur_output_fn in zip(labels, filenames):
            f.write(cur_output_fn + '\t' + cur_output_gt + '\n') 