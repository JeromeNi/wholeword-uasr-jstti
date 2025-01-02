import os
import json

## Applies the given file splits to the LibriSpeech clean pruned dataset
## Make changes to `topk`, `out_librispeech_dir`, and `train/valid/valid_true_file_list_fn` to match your setup

def get_file_set(file_list_fn):
    file_list = []
    with open(file_list_fn, 'r') as f:
        lines = f.readlines()
    for l in lines:
        file_list.append(l.strip())
        
    file_list = set(file_list)
    return file_list

topk = 4096

out_librispeech_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'

train_file_list_fn = '/path/to/train_file_list.txt'

valid_file_list_fn = '/path/to/valid_file_list.txt'

valid_true_file_list_fn = '/path/to/valid_true_file_list.txt'

train_file_list = get_file_set(train_file_list_fn)

valid_file_list = get_file_set(valid_file_list_fn)

valid_true_file_list = get_file_set(valid_true_file_list_fn)

boundary_json = os.path.join(out_librispeech_dir, f'ls_train_clean_pruned_top{topk}_coco_style_no_sil.json')
with open(boundary_json, 'r') as f:
    data_dict = json.load(f)['data']


train_entries = []
valid_entries = []
valid_true_entries = []

for data_entry in data_dict:
    if data_entry['caption']['wav'] in train_file_list:
        train_entries.append(data_entry)
    if data_entry['caption']['wav'] in valid_file_list:
        valid_entries.append(data_entry)
    if data_entry['caption']['wav'] in valid_true_file_list:
        valid_true_entries.append(data_entry)
        
print(len(train_entries), len(valid_entries), len(valid_true_entries))

with open(os.path.join(out_librispeech_dir, f'ls_train_clean_pruned_top{topk}_coco_style_no_sil_train.json'), 'w') as f:
    json.dump({"data": train_entries}, f)
with open(os.path.join(out_librispeech_dir, f'ls_train_clean_pruned_top{topk}_coco_style_no_sil_valid.json'), 'w') as f:
    json.dump({"data": valid_entries}, f)
with open(os.path.join(out_librispeech_dir, f'ls_train_clean_pruned_top{topk}_coco_style_no_sil_valid_true.json'), 'w') as f:
    json.dump({"data": valid_true_entries}, f)