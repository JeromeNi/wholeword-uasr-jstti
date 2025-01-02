import os
import shutil

# point to the directory containing the checkpoint, together with the evaluation results for all three splits (within the folder called `predictions_with_fn`)
topk=2048
feature_type='xxx' # be consistent with the naming of the features
wrd_bnd_type='your_flag_for_bnd_used' # be consistent with the naming of the word

# point to your data directory for a specific boundary type
datadir=f'/path/to/data/libri-train-clean-top-{topk}/feat_{feature_type}_no_sil_{wrd_bnd_type}/top_{topk}/km_dir_{topk}'
CKPT_PATH=f'{datadir}/your_checkpoint_folder_name'

# point to the original tsv files prepared when curating the topk-word corpus
TSV_PATH=f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/tsvs'

# whole-word text dictionary
DICT_PATH=f'{datadir}/dict.txt'


self_train_tsvs_dir = os.path.join(CKPT_PATH, 'self_train_tsvs')
os.makedirs(self_train_tsvs_dir,exist_ok=True)

gt_pred_dir = os.path.join(CKPT_PATH, 'predictions_with_fn')

for which_set in ["train", "valid", "valid_true"]:
    tsv_file = os.path.join(TSV_PATH, which_set + '.tsv')
    with open(tsv_file, 'r') as f:
        tsv_lines = f.readlines()
        
    parent_dir = tsv_lines[0]
    tsv_dict = {}
    for tsv_line in tsv_lines[1:]:
        tsv_line_splitted = tsv_line.split('\t')
        tsv_dict[tsv_line_splitted[0]] = tsv_line_splitted[1]
        
    preds_fn = os.path.join(gt_pred_dir, f'{which_set}_layer_0_preds.txt')
    preds_dict = {}
    with open(preds_fn, 'r') as f:
        pred_lines = f.readlines()
        for pred_line in pred_lines:
            cur_fn, cur_pred = pred_line.split('\t')
            preds_dict[cur_fn] = cur_pred
    
    ref_fn = os.path.join(gt_pred_dir, f'{which_set}_layer_0_ref.txt')
    ref_dict = {}
    with open(ref_fn, 'r') as f:
        ref_lines = f.readlines()
        for ref_line in ref_lines:
            cur_fn, cur_ref = ref_line.split('\t')
            ref_dict[cur_fn] = cur_ref
    
    print(len(ref_dict), len(preds_dict))
    
    with open(os.path.join(self_train_tsvs_dir, f'{which_set}.tsv'), 'w') as f_tsv, open(os.path.join(self_train_tsvs_dir, f'{which_set}_gt.tsv'), 'w') as f_tsv_gt, open(os.path.join(self_train_tsvs_dir, f'{which_set}.wrd'), 'w') as f_wrd, open(os.path.join(self_train_tsvs_dir, f'{which_set}_gt.wrd'), 'w') as f_wrd_gt:
        f_tsv.write(parent_dir)
        f_tsv_gt.write(parent_dir)
        for key in ref_dict.keys():
            f_tsv.write("\t".join([key, tsv_dict[key]]))
            f_tsv_gt.write("\t".join([key, tsv_dict[key]]))
            f_wrd.write(preds_dict[key])
            f_wrd_gt.write(ref_dict[key])
    
    
    
shutil.copy(src=DICT_PATH,dst=os.path.join(self_train_tsvs_dir, 'dict.wrd.txt'))