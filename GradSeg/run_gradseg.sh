CONDA_ROOT=/path/to/anaconda3

## Activate irtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh

conda activate gradseg
export LD_LIBRARY_PATH=${CONDA_ROOT}/envs/gradseg/lib/:$LD_LIBRARY_PATH

# hyperparameters for 1024-word corpus
topk=1024
audio_dir=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil
python grad_segmenter_new_loader.py --min_separation 3 --reg 1e7 --target_perc 40 --frames_per_word 12 --train_tsv ${audio_dir}/tsv/speech_train_100utts.tsv --val_tsv ${audio_dir}/tsvs/train_valid_true.tsv --train_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train.json --val_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train_valid_true.json --save_bounds_dir ${audio_dir}/gradseg_unsup_segmentation/ --save_name train_valid_true_utts_testing

# hyperparameters for 2048-word corpus
topk=2048
audio_dir=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil
python grad_segmenter_new_loader.py --min_separation 1 --reg 5e7 --target_perc 40 --frames_per_word 13 --train_tsv ${audio_dir}/tsvs/train_100utts.tsv --val_tsv ${audio_dir}/tsvs/train_valid_true.tsv --train_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train.json --val_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train_valid_true.json --save_bounds_dir ${audio_dir}/gradseg_unsup_segmentation/ --save_name train_valid_true_utts_testing

# hyperparameters for 4096-word corpus
topk=4096
audio_dir=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil
python grad_segmenter_new_loader.py --min_separation 1 --reg 9e8 --target_perc 40 --frames_per_word 14 --train_tsv $audio_dir/tsvs/train_100utts.tsv --val_tsv $audio_dir/tsvs/train_valid_true.tsv --train_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train.json --val_boundary_json ${audio_dir}/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_train_valid_true.json --save_bounds_dir ${audio_dir}/gradseg_unsup_segmentation/ --save_name train_valid_true_utts_testing

echo "Run completed at:- "
date