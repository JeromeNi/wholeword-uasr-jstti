import os

## convert boundary format
import pickle
import numpy as np

topk=2048
audio_dir=f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/'
for which_set in ['train', 'valid', 'valid_true']:
    gradseg_file = f'{audio_dir}/gradseg_unsup_segmentation/{which_set}_utts.pkl'
    write_gt_file = f'{audio_dir}/gt_sup_segmentation/{which_set}_utts.pkl'
    os.makedirs(os.path.dirname(write_gt_file), exist_ok = True)
        
    with open(gradseg_file, 'rb') as fb:
        gradseg_bnd_dict = pickle.load(fb)
        
    new_gt_bnd_dict = {}
        
    file_keys = list(gradseg_bnd_dict.keys())

    for file_key in file_keys:
        item_dict = {}
        item_dict['ref_bound'] = np.copy(gradseg_bnd_dict[file_key]['ref_bound'])
        item_dict['seg_bound'] = np.copy(gradseg_bnd_dict[file_key]['ref_bound'])
        
        new_gt_bnd_dict[file_key] = item_dict
    
    with open(write_gt_file, 'wb') as fwb:
        pickle.dump(new_gt_bnd_dict, fwb)