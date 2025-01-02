import pickle
import numpy as np
import os
import torch
import math

# Vocab size
topk = 2048
# Modify to the directory where your pruned corpus is stored
parent_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/'

# point to where the tsv files for the pruned corpus are stored.
tsv_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/tsvs/'

# where to save the generated files for the topk word corpus for training wav2boundaries
write_dir = f'/path/to/wav2boundaries/librispeech-clean-no-sil-top-{topk}/'

# point to where you stored the downloaded file lists. These are used to filter out bad files.
file_dir = '/path/to/downloaded/file_lists'

for which_set in ['train', 'valid', 'valid_true']:

    with open(os.path.join(file_dir, f'{which_set}_file_list.txt'), 'r') as f:
        all_file_lines = f.readlines()
    print(len(all_file_lines))
    with open(os.path.join(file_dir, f'{which_set}_bnd_errors.txt'), 'r') as f:
        bad_file_lines = f.readlines()

    bad_files = set([f.strip().split()[0] for f in bad_file_lines])
    print(bad_files)
    selected_files = [f.strip() for f in all_file_lines if f.strip() not in bad_files]
    print(len(selected_files), selected_files[:10])

    vad_dir = os.path.join(write_dir, 'vads')
    os.makedirs(vad_dir, exist_ok = True)


    write_vad_file = os.path.join(vad_dir, f'{which_set}.vad')
    tsv_fn = os.path.join(tsv_dir, f'{which_set}.tsv')

    with open(tsv_fn, 'r') as f:
        f_lines = f.readlines()[1:]
        
    print(bad_files)

    with open(write_vad_file, 'w') as fw:
        for f_line in f_lines:
            fn, num_frames = f_line.strip().split('\t')
            if fn not in bad_files:
                # for librispeech naming only
                spk = fn.split('/')[1]
                end_in_sec = float(num_frames) / 16000
                end_in_sec = math.floor(end_in_sec * 100)/100.0
                fw.write(os.path.join(parent_dir, fn) + ' ' + spk + ' ' + '0' + ' ' + str(end_in_sec) + '\n')
            else:
                print(fn, num_frames)
        