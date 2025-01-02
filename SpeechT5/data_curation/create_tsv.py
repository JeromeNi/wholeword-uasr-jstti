## This script generates a tsv file for the pruned LibriSpeech dataset with topk words
## Make changes to `topk`, `out_librispeech_dir`, and `train/valid/valid_true_file_list_fn` to match your setup



import numpy as np
import os
import json
import soundfile as sf
import copy

def get_file_set(file_list_fn):
    file_list = []
    with open(file_list_fn, 'r') as f:
        lines = f.readlines()
    for l in lines:
        file_list.append(l.strip())
        
    return file_list


def process_tsv(parent_dir, save_dir, file_list, which_set):
    print(which_set, len(file_list))
    with open(os.path.join(save_dir, f'{which_set}.tsv'), 'w') as fw:
        fw.write(parent_dir + '\n')
        for fn in file_list:
            fw.write(fn + '\t' + str(sf.info(os.path.join(parent_dir, fn)).frames) + '\n')

topk = 4096
out_librispeech_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'

train_file_list_fn = '/path/to/train_file_list.txt'

valid_file_list_fn = '/path/to/valid_file_list.txt'

valid_true_file_list_fn = '/path/to/valid_true_file_list.txt'

train_file_list = get_file_set(train_file_list_fn)

valid_file_list = get_file_set(valid_file_list_fn)

valid_true_file_list = get_file_set(valid_true_file_list_fn)

os.makedirs(os.path.join(out_librispeech_dir, 'tsvs'),exist_ok=True)

process_tsv(out_librispeech_dir, os.path.join(out_librispeech_dir, 'tsvs'), train_file_list, 'train')
process_tsv(out_librispeech_dir, os.path.join(out_librispeech_dir, 'tsvs'), valid_file_list, 'valid')
process_tsv(out_librispeech_dir, os.path.join(out_librispeech_dir, 'tsvs'), valid_true_file_list, 'valid_true')





