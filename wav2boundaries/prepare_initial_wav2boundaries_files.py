import pickle
import numpy as np
import os
import torch
import math

# Vocab size
topk = 2048
# Modify to the directory where your pruned corpus is stored
parent_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/'

# Do not need to modify
which_set = 'train'

# where to save the generated files for the topk word corpus for training wav2boundaries
write_dir = f'/path/to/wav2boundaries/librispeech-clean-no-sil-top-{topk}/'

# give the experiment some meaningful name
bnd_dir = os.path.join(write_dir, 'hubert_gradseg')

os.makedirs(bnd_dir, exist_ok = True)

# Modify to the directory where your word segmentation (say, from GradSeg) is stored. These will be the labels for wav2boundaries training
manifest_path = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/gradseg_unsup_segmentation/'

# point to where you stored the downloaded file lists. These are used to filter out bad files.
file_dir = '/path/to/downloaded/file_lists'

with open(os.path.join(file_dir, f'{which_set}_file_list.txt'), 'r') as f:
    all_file_lines = f.readlines()
print(len(all_file_lines))
with open(os.path.join(file_dir, f'{which_set}_bnd_errors.txt'), 'r') as f:
    bad_file_lines = f.readlines()

bad_files = set([f.strip().split()[0] for f in bad_file_lines])
print(bad_files)
selected_files = [f.strip() for f in all_file_lines if f.strip() not in bad_files]
print(len(selected_files), selected_files[:10])



with open(os.path.join(manifest_path, f'{which_set}_utts.pkl'), 'rb') as fb:
    boundary_dict = pickle.load(fb)

    
with open(os.path.join(bnd_dir, which_set), 'w') as fw:

    for filename in selected_files:
        boundaries = boundary_dict[filename]
        
        spk = filename.split('/')[1]
        
        boundaries['seg_bound'][-1] = 1
        boundary_locs = list(np.where(boundaries['seg_bound'] == 1)[0])

        if boundary_locs[0] != 0:
            boundary_locs = [0] + boundary_locs
            
        fw.write(os.path.join(parent_dir, filename) + ' ' + spk + ' ')
        word_list = []
        for start, end in zip(boundary_locs[:-1], boundary_locs[1:]):
            start_in_sec = start * 0.01
            start_in_sec = math.floor(start_in_sec * 100)/100.0
            end_in_sec = end * 0.01
            end_in_sec = math.floor(end_in_sec * 100)/100.0
            word_list.append(str(start_in_sec) + "," + str(end_in_sec))
            
        fw.write(' '.join(word_list) + '\n')
        