# Converts VG-HuBERT word boundaries to the pkl format of GradSeg output, which is used by the dataloaders of the JSTTI models
# Modify `gradseg_file` and `vghubert_file` with the paths to your GradSeg and VG-HuBERT outputs, respectively. Modify `write_vghubert_file` as the save path of the converted file.
# If you have manually inspect the VG-HuBERT boundaries and found some null outputs, you can use `direct_replace` to replace them with the output of GradSeg.

import os

import pickle
import numpy as np

topk = 4096
for which_set in ['train', 'valid', 'valid_true']:
    gradseg_file = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/gradseg_unsup_segmentation/{which_set}_utts.pkl'
    vghubert_file = f'/path/to/vg_hubert_feats/vg-hubert_3_9_4096_0.7_mean_clsAttnBnd_1_librispeech_clean_pruned_top{topk}_no_sil_{which_set}_set/vg-hubert_3/librispeech_clean_pruned_top{topk}_no_sil_{which_set}_set_mean_0.7_9_clsAttnBnd/data_dict.pkl'
    write_vghubert_file = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/vghubert_unsup_segmentation/{which_set}_utts.pkl'
    os.makedirs(os.path.dirname(write_vghubert_file), exist_ok = True)

    with open(vghubert_file, 'rb') as fb:
        vghubert_feat_dict = pickle.load(fb)
        

    with open(gradseg_file, 'rb') as fb:
        gradseg_bnd_dict = pickle.load(fb)
        
    new_vghubert_bnd_dict = {}
        
    file_keys = list(gradseg_bnd_dict.keys())
    print(vghubert_feat_dict[file_keys[0]]['boundaries'])
    
    # direct_replace = ['train-clean-360/1806/143948/1806-143948-0024.flac', 'train-clean-360/7717/104491/7717-104491-0018.flac']
    direct_replace = []
    
    for file_key in file_keys:
        item_dict = {}
        item_dict['ref_bound'] = np.copy(gradseg_bnd_dict[file_key]['ref_bound'])
        item_dict['seg_bound'] = np.zeros_like(gradseg_bnd_dict[file_key]['seg_bound'])
        if file_key in direct_replace:
            item_dict['seg_bound'] = gradseg_bnd_dict[file_key]['seg_bound']
            new_vghubert_bnd_dict[file_key] = item_dict
            print(file_key)
            continue
        spf = vghubert_feat_dict[file_key]['spf']
        bad_bnd = 0
        if 'boundaries_frame' not in vghubert_feat_dict[file_key]:
            boundary_iter = [(round((boundary_start / spf) * 2), round((boundary_end / spf) * 2)) for boundary_start, boundary_end in vghubert_feat_dict[file_key]['boundaries'].tolist()]
        else:
            boundary_iter = [(int(boundary_start * 2), int(boundary_end * 2)) for boundary_start, boundary_end in vghubert_feat_dict[file_key]['boundaries_frame'].tolist()]
            # boundary_iter_comp = [(round((boundary_start / spf) * 2), round((boundary_end / spf) * 2)) for boundary_start, boundary_end in vghubert_feat_dict[file_key]['boundaries'].tolist()]
            # print(boundary_iter)
            # print(boundary_iter_comp)
        for rounded_start, rounded_end in boundary_iter:
            if rounded_start < len(item_dict['seg_bound']):
                item_dict['seg_bound'][rounded_start] = 1
            else:
                bad_bnd += 1
            if rounded_end < len(item_dict['seg_bound']):
                item_dict['seg_bound'][rounded_end] = 1
            else:
                bad_bnd += 1
            
        if bad_bnd > 1:
            print(file_key, bad_bnd)
        
        new_vghubert_bnd_dict[file_key] = item_dict
        

    with open(write_vghubert_file, 'wb') as fwb:
        pickle.dump(new_vghubert_bnd_dict, fwb)