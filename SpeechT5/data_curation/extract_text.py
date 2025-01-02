import numpy as np
import shutil
import os
import json

# This scripts writes the transcripts for each subset of LibriSpeech clean pruned dataset into separate text files for later processing into fairseq's .bin/.idx files
# Make sure to replace `/path/to` in `write_dir`, `file_list_fn`, `meta_data_json_fn` with your actual paths


# Additional Note: In my file structure, I have some data folder that separates the experiments on different topk=1024/2048/4096 into different folders (`libri-train-clean-top-{topk}``); within the folder `libri-train-clean-top-{topk}`, I separate different experiments obtained with different word boundaries and different feature types (mostly, just `hubert_large_l21_ori_feats`) into `feat_{feature_type}_no_sil_{wrd_bnd_type}`. Within this is a single folder called `top_{topk}`, which stores the text files for each subset, and a subfolder called `km_dir_{topk}` to store actual experiment data and checkpoints.
'''
 ```
    data
    └── libri-train-clean-top-{topk}
        └── feat_{feature_type}_no_sil_{wrd_bnd_type}
            └── top_{topk}
                ├── km_dir_{topk}
                |    ├──discrete_speech_train_clsattnbndkmeans
                |    ├──discrete_speech_valid_clsattnbndkmeans
                |    ├──discrete_speech_valid_true_clsattnbndkmeans
                |    ├──text_train
                |    ├──text_valid
                |    ├──text_valid_true
                |    ├──feats_for_l1
                |    └──text_for_l1
                ├── train_words.txt
                ├── valid_words.txt
                └── valid_true_words.txt
    ```
'''

topk = 4096
write_dir = f'/path/to/libri-train-clean-top-{topk}/feat_hubert_large_l21_ori_feats_no_sil_gt_bnd/top_{topk}'
os.makedirs(os.path.join(write_dir, f'km_dir_{topk}'), exist_ok =True)


for which_set in ['train', 'valid', 'valid_true']:
    file_list_fn = f'/path/to/{which_set}_file_list.txt'
    meta_data_json_fn = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/ls_train_clean_pruned_top{topk}_gradseg_style_no_sil_{which_set}.json'
    
    with open(file_list_fn, 'r') as f:
        file_lines = f.readlines()
        
    with open(meta_data_json_fn, 'r') as f:
        meta_data_dict = json.load(f)
    
    with open(os.path.join(write_dir, f'{which_set}_words.txt'), 'w') as fw:
        for file_line in file_lines:
            file_line = file_line.strip()
            fw.write(" ".join([c[2] for c in meta_data_dict[file_line]]) + '\n')
            

        
shutil.copy(f'dict_{topk}.txt', os.path.join(write_dir,'dict.txt'))  
shutil.copy(f'dict_{topk}.txt', os.path.join(write_dir,f'km_dir_{topk}','dict.txt'))  
    
    