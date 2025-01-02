# This script creates librispeech-clean-pruned with topk words
# Choose `topk` between 1024/2048/4096, and replace `boundary_json`, `in_librispeech_dir`, and `out_librispeech_dir` with your own paths.

import numpy as np
import os
import json
import soundfile as sf
import copy

# Vocabulary file for topk words
topk = 4096
topk_dict_fn = f'dict_{topk}.txt'

with open(topk_dict_fn, 'r') as f:
    lines = f.readlines()
    
top_k_dict = {}

for l in lines:
    splitted_l = l.strip().split()
    top_k_dict[splitted_l[0]] = int(splitted_l[1])

boundary_json = '/path/to/ls_train_clean_coco_style.json'
with open(boundary_json, 'r') as f:
    data_dict = json.load(f)
    
print(data_dict['data'][0])

# path to LibriSpeech dataset and output directory
in_librispeech_dir = '/path/to/LibriSpeech'
out_librispeech_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'

bad_check = 0
removed_utts = 0
new_data_dict = []
for idx, item in enumerate(data_dict['data']):
    if (idx + 1) % 1000 == 0:
        print(f'Processed {idx + 1} files')
    alignment = item['text_alignment']
    filename = os.path.join(in_librispeech_dir, item['caption']['wav'])
    new_filename = os.path.join(out_librispeech_dir, item['caption']['wav'])
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)
    
    data, samplerate = sf.read(filename)
    
    data_list = []
    
    new_word_ali = []
    prev_total_time = 0
    prev_end_time = 0
    for word_idx, word_ali in enumerate(alignment.split()):
        start_time, word, end_time = word_ali.split('__')
        start_time = float(start_time)
        end_time = float(end_time)
        
        if word in top_k_dict:
            if word_idx == 0:
                data_list.append(data[int(start_time * samplerate): int(end_time * samplerate)])
            else:
                if int(prev_end_time * samplerate) - int(start_time * samplerate) < 0:
                    prev_total_time += min(0.04, (start_time - prev_end_time))
                    sil_dur = min(0.04, (start_time - prev_end_time))
                    # prev_total_time += (start_time - prev_end_time)
                    data_list.append(data[int(prev_end_time * samplerate): int(((sil_dur/2) + prev_end_time) * samplerate)])
                    data_list.append(data[int((-(sil_dur/2) + start_time)* samplerate): int(start_time * samplerate)])
                data_list.append(data[int(start_time * samplerate): int(end_time * samplerate)])
            new_start_time = prev_total_time
            new_end_time = prev_total_time + (end_time - start_time)
            prev_total_time += (end_time - start_time)
            new_word_ali.append([new_start_time, word, new_end_time])
            
        prev_end_time = end_time
    
    if len(new_word_ali) > 0:
        new_item = copy.deepcopy(item)
        new_item['caption']['text'] = ' '.join([w for s,w,e in new_word_ali])
        new_item['text_alignment'] = ' '.join(["__".join([str(s),w,str(e)]) for s,w,e in new_word_ali])

        new_data_dict.append(new_item)

        new_data = np.concatenate(data_list,axis = 0)
        if np.abs(len(new_data) - samplerate * new_word_ali[-1][-1]) > 10:
            bad_check += 1
            print('bad element where the end is off by 10 samples or more:', len(new_data) ,samplerate * new_word_ali[-1][-1])
        sf.write(new_filename, new_data, samplerate)
    else:
        removed_utts += 1
        
print('bad check where the end is off by 10 samples or more:', bad_check)
print('example data element:', new_data_dict[0])

new_boundary_json = f'{out_librispeech_dir}/ls_train_clean_pruned_top{topk}_coco_style_no_sil.json'
with open(new_boundary_json, 'w') as f:
    save_data_dict = {}
    save_data_dict['data'] = new_data_dict
    json.dump(save_data_dict, f)
    
    
print('total number of utterances and total utterances removed:', len(new_data_dict), removed_utts)