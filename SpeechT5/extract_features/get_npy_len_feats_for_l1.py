# create dataset for l1 gan
import os
import numpy as np

for which_set in ['train', 'valid', 'valid_true']:
    topk=2048
    feature_type = '' # be consistent with the naming of the features
    wrd_bnd_type = '' # be consistent with the naming of the word boundaries
    # point to the directory structure in step 8
    datadir = f'/path/to/data/libri-train-clean-top-{topk}/feat_{feature_type}_no_sil_{wrd_bnd_type}/top_{topk}/km_dir_{topk}/'


    km_file = f'{datadir}/seg_feats/km_idx_clsattnbndkmeans/{which_set}.km'
    feat_dir = os.path.join(datadir, 'feats_for_l1_clsattnbndkmeans')
    os.makedirs(os.path.join(datadir, 'feats_for_l1_clsattnbndkmeans'), exist_ok = True)

    save_arr = []
    save_len = []

    with open(km_file, 'r') as f:
        lines = f.readlines()
        
        for l in lines:
            splitted_l = l.strip().split()
            splitted_l = [int(c) for c in splitted_l]
            save_arr.extend(splitted_l)
            save_len.append(len(splitted_l))
            
    with open(os.path.join(feat_dir, f'{which_set}.lengths'), 'w') as f:
        for length in save_len:
            f.write(str(length) + '\n')

    print(len(save_len))
    save_arr = np.expand_dims(np.array(save_arr), axis = -1)
    print(save_arr.shape)
    np.save(os.path.join(feat_dir, f'{which_set}.npy'), save_arr)