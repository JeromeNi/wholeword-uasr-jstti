import numpy as np
import os
import json

topk= 4096 # or 1024 or 2048
audio_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'
tsv_fns = [f'{audio_dir}/tsvs/train.tsv', f'{audio_dir}/tsvs/valid_true.tsv']

dict_fns = [f'{audio_dir}/ls_train_clean_pruned_top{topk}_gradseg_style_no_sil_train.json', f'{audio_dir}/ls_train_clean_pruned_top{topk}_gradseg_style_no_sil_valid_true.json']

all_dict = {}

write_tsv_fn = f'{audio_dir}/tsvs/train_valid_true.tsv'
write_dict_fn = f'{audio_dir}/ls_train_clean_pruned_top4096_gradseg_style_no_sil_train_valid_true.json'

parent_dir = None
join = False
for tsv_fn in tsv_fns:
    with open(tsv_fn, 'r') as f:
        line = f.readlines()[0]
        
    if parent_dir is not None and line != parent_dir:
        parent_dir = '/' + '\n'
        join = True
    else:
        parent_dir = line
        
with open(write_tsv_fn, 'w') as fw:
    fw.write(parent_dir)
    for tsv_fn, dict_fn in zip(tsv_fns, dict_fns):
        with open(tsv_fn, 'r') as f:
            lines = f.readlines()
        cur_pd = lines[0].strip()
        files = lines[1:]
        
        for f in files:
            if join:
                write_fn = os.path.join(cur_pd, f.split('\t')[0]) + f.split('\t')[1]
            else:
                write_fn = f
        
            fw.write(write_fn)
        
        
        with open(dict_fn, 'r') as f:
            cur_dict = json.load(f)
        for k, v in cur_dict.items():
            all_dict[k] = v
            
with open(write_dict_fn, 'w') as fw:
    json.dump(all_dict, fw)       
            
        