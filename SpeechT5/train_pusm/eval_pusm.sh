# copy from the train_pusm_xxxx.sh

topk=1024
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_word_seg_bnds # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
tgt_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/

TASK_DATA=$tgt_dir/feats_for_l1_clsattnbndkmeans

# where to save the checkpoints
ckpt_dir=$tgt_dir/multirun/l1_w2vu_clsattnkmeans_valid_with_aggregated_stats


DATA_DIR=$TASK_DATA
CKPT_PATH=$ckpt_dir/0/checkpoint_best.pt
echo $CKPT_PATH

USER_DIR=/path/to/wav2vecu_word_small_batch/

python get_wav2vecu_word_preds_mod.py $DATA_DIR --split "valid_true" --ckpt-path $CKPT_PATH --user-dir ${USER_DIR}
