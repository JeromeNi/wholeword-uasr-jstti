import numpy as np
import os
from npy_append_array import NpyAppendArray
import math
import torch.nn.functional as F
import json
import pickle

for which_set in ['train', 'valid', 'valid_true']:
    topk=2048
    feature_type = '' # be consistent with the naming of the features
    wrd_bnd_type = '' # be consistent with the naming of the word boundaries
    # point to the output of running the vg_hubert_feats script
    vghubert_file = f'/path/to/vg_hubert_feats/vg-hubert_3_9_4096_0.7_mean_clsAttnBnd_1_librispeech_clean_pruned_top{topk}_no_sil_{which_set}_set/vg-hubert_3/librispeech_clean_pruned_top{topk}_no_sil_{which_set}_set_mean_0.7_9_clsAttnBnd/data_dict.pkl'
    # point to the directory structure in step 8
    data_path = f'/path/to/data/libri-train-clean-top-{topk}/feat_{feature_type}_no_sil_{wrd_bnd_type}/top_{topk}/km_dir_{topk}/'


    manifest_path = os.path.join(data_path, f'discrete_speech_{which_set}_clsattnbndkmeans')
    data = np.load(os.path.join(manifest_path, 'data.npy'), mmap_mode='r')

    with open(os.path.join(manifest_path, 'data.lengths'), 'r') as f:
        data_lens_f = f.readlines()

    data_lens = []

    for line in data_lens_f:
        data_lens.append(int(line.strip()))
        
    offsets = [0] + list(np.cumsum(data_lens)[:-1])

    filenames = []
    with open(os.path.join(manifest_path, 'file_list.txt'), 'r') as f:
        file_lines = f.readlines()
    for file_line in file_lines:
        filenames.append(file_line.strip())

    with open(vghubert_file, 'rb') as fb:
        vghubert_feat_dict = pickle.load(fb)

    seg_feats_path = os.path.join(data_path, 'seg_feats')

    os.makedirs(seg_feats_path, exist_ok=True)
    if os.path.exists(os.path.join(seg_feats_path, f'{which_set}_0_1.npy')):
        os.remove(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))
    npaa = NpyAppendArray(os.path.join(seg_feats_path, f'{which_set}_0_1.npy'))


    with open(os.path.join(seg_feats_path, f'{which_set}_0_1.len'), 'w') as fw:
        for idx, file_key in enumerate(filenames):

            curr_data = data[offsets[idx]: offsets[idx] + data_lens[idx]]
            curr_seg_feats = []
            # print(vghubert_feat_dict[file_key])
            spf = vghubert_feat_dict[file_key]['spf']
            
            if 'boundaries_frame' not in vghubert_feat_dict[file_key]:
                for boundary_start, boundary_end in vghubert_feat_dict[file_key]['boundaries'].tolist():
                    if ((boundary_start / spf) - int(boundary_start / spf)) >= 0.6:
                        rounded_start = int(boundary_start / spf) + 1
                        # print(boundary_start / spf)
                    else:
                        rounded_start = int(boundary_start / spf)
                    if (boundary_end / spf - int(boundary_end / spf)) >= 0.6:
                        rounded_end = int(boundary_end / spf) + 1
                        # print(boundary_end / spf)
                    else:
                        rounded_end = int(boundary_end / spf)
                    curr_seg_data = np.mean(curr_data[rounded_start: rounded_end, :], axis = 0)
                    curr_seg_feats.append(curr_seg_data)
            else:
                for rounded_start, rounded_end in vghubert_feat_dict[file_key]['boundaries_frame'].tolist():
                    curr_seg_data = np.mean(curr_data[int(rounded_start): int(rounded_end), :], axis = 0)
                    curr_seg_feats.append(curr_seg_data)

            curr_seg_feats = np.stack(curr_seg_feats, axis = 0)
            npaa.append(curr_seg_feats)
            print(len(curr_seg_feats), file=fw)


