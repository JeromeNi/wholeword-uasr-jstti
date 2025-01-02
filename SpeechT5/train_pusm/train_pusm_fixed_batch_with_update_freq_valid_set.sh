# The following script is originally designed to run on a slurm-based server; you will need to modify to match your training setup

PREFIX=w2v_unsup_gan_xp

topk=1024
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_word_seg_bnds # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
tgt_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/
# config file used
CONFIG_NAME=l1_w2vu_top1024_kmdir1024_fixed_size_batch_valid_update_freq_18

TASK_DATA=$tgt_dir/feats_for_l1_clsattnbndkmeans
TEXT_DATA=$tgt_dir/text_for_l1

KENLM_PATH=$tgt_dir/text_for_l1/kenlm.wrd.o40003.bin  # KenLM 4-gram word language model

# where to save the checkpoints
ckpt_dir=$tgt_dir/multirun/l1_w2vu_clsattnkmeans_valid_no_aggregated_stats_with_update_freq_18


PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX srun --ntasks=1 --exclusive --gres=gpu:1 --mem=240G -c 20 fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name $CONFIG_NAME \
    task.data=$TASK_DATA \
    task.text_data=$TEXT_DATA \
    task.kenlm_path=$KENLM_PATH \
    common.user_dir=$(pwd)/../wav2vecu_word \
    model.code_penalty=0 model.gradient_penalty=0.0 \
    model.smoothness_weight=0.0 'common.seed=range(0,1)' \
    checkpoint.save_dir='./' \
    hydra.run.dir=$ckpt_dir \
    hydra.sweep.dir=$ckpt_dir 