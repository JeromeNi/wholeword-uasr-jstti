import pickle
import os
import numpy as np

topk=4096
audio_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'
for which_set in ['train', 'valid', 'valid_true']:
    
    error_fns = []
    bnd_errors_fn = os.path.join(f'/path/to/{which_set}_bnd_errors.txt')
    
    with open(bnd_errors_fn, 'r') as f:
        error_fns = set([l.strip().split(maxsplit = 1)[0] for l in f.readlines()])
        
    manifest_path = f'{audio_dir}/vghubert_unsup_segmentation/'
    with open(os.path.join(manifest_path, f'{which_set}_utts.pkl'), 'rb') as fb:
        boundary_dict = pickle.load(fb)
        
    for filename in boundary_dict:
        ref_bound = boundary_dict[filename]['ref_bound']
        seg_bound = boundary_dict[filename]['seg_bound']

        boundary_locs = np.where(seg_bound == 1)[0]
        f_err = False
        if len(boundary_locs) == 0:
            boundary_locs = np.arange(0, len(ref_bound), 26).astype('int32')
            f_err = True
            if filename not in error_fns:
                print(which_set, filename, 1)
        elif len(boundary_locs) == 1:
            boundary_locs = np.arange(0, len(ref_bound), 26).astype('int32')
            f_err = True
            if filename not in error_fns:
                print(which_set, filename, 2)
        elif len(boundary_locs) == 1:
            boundary_locs = [0, len(ref_bound)]
            f_err = True
            if filename not in error_fns:
                print(which_set, filename, 3)

