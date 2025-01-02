## Splits the GradSeg output into train, valid, and valid_true
## Modify `audio_dir` with the audio directory, `save_utts_dir` with the path to the output of run_gradseg.sh
import pickle
import json
import os

topk=4096
audio_dir = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil'
save_utts_dir = f'{audio_dir}/gradseg_unsup_segmentation/'
all_utts_fn = f'{save_utts_dir}/train_valid_true_utts_testing.pkl'
all_json_file = f'{audio_dir}/ls_train_clean_pruned_top{topk}_gradseg_style_no_sil_'

which_sets = ['train', 'valid', 'valid_true']

with open(all_utts_fn, 'rb') as fb:
    all_utts_dict = pickle.load(fb)
    
    
for which_set in which_sets:
    curr_json_file = all_json_file + f'{which_set}.json'
    with open(curr_json_file, 'r') as f:
        utts_keys = list(json.load(f).keys())
    
    print(which_set, len(utts_keys))
    with open(os.path.join(save_utts_dir, f'{which_set}_utts.pkl'), 'wb') as fwb:
        pickle.dump({k: all_utts_dict[k] for k in utts_keys}, fwb)
