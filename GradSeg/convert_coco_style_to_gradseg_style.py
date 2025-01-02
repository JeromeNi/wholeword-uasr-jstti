import os

## convert boundary format
import pickle
import numpy as np

import json

topk = 4096
audio_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'
for which_set in ['train', 'valid', 'valid_true']:
    input_fn = f'{audio_dir}/ls_train_clean_pruned_top{topk}_coco_style_no_sil_{which_set}.json'
    output_fn = f'{audio_dir}/ls_train_clean_pruned_top{topk}_gradseg_style_no_sil_{which_set}.json'


    output_dict = {}
    with open(input_fn, 'r') as f:
        all_data_coco = json.load(f)['data']
        
    for data_entry in all_data_coco:
        wav_fn = data_entry["caption"]["wav"]
        text_alignment = data_entry["text_alignment"]
        output_dict[wav_fn] = []
        for ali in text_alignment.split():
            start_time, word, end_time = ali.split("__")
            start_sample = int(16000 * float(start_time))
            end_sample = int(16000 * float(end_time))
            
            output_dict[wav_fn].append([start_sample, end_sample, word]) 
            
    with open(output_fn, 'w') as fw:
        json.dump(output_dict, fw)