# The following script is originally designed to run on a slurm-based server,
# configured as a 2-node setup, where each node has 4 GPUs. You will need to change
# the script to adapt to your distributed training environment!

DISTRIBUTED_WORLD_SIZE=8
DISTRIBUTED_PORT=12345
USER_DIR=/path/to/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024_small_to_large/speecht5

topk=4096
small_size=1024
feature_type=xxx # be consistent with the naming of the features
FEAT_DIM=1024 # the dimension of HuBERT-large features
wrd_bnd_type=your_flag_for_word_seg_bnds_on_wav2bnd_large_1024 # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
DATA_ROOT=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}
# where to save the model
SAVE_DIR=$DATA_ROOT/model_codebook_0.3_clsattn_bnd_kmeans_encoder_only_retry_2024_2l_no_diff_bnd_from_1024

# Point to the directory containing the training and validation data (step 8)
TRAIN_SET="discrete_speech_train_clsattnbndkmeans|text_train"
VALID_SET="discrete_speech_valid_clsattnbndkmeans|text_valid"

# note that INIT_CLUS_DIR points to the km model obtained from FC-kmeans (with the first 1024 clusters fixed)
INIT_CLUS_DIR=${DATA_ROOT}/seg_feats/km_model_fckmeans_from_${small_size}.npy

# point to the previosly trained best fully unsupervised JSTTI model
wrd_bnd_type_for_wav2bnd_large_bnds=your_flag_for_wav2bnd_large_trained_on_jsttie2e_bnds
INIT_MODEL_DIR=/path/to/data/libri-train-clean-top-${small_size}/feat_${feature_type}_no_sil_${wrd_bnd_type_for_wav2bnd_large_bnds}/top_${small_size}/km_dir_${small_size}/model_codebook_0.3_clsattn_bnd_kmeans_encoder_only_retry_2024_2l_no_diff_bnd/checkpoint_best.pt


srun fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --distributed-world-size ${DISTRIBUTED_WORLD_SIZE} \
  --distributed-port ${DISTRIBUTED_PORT} \
  --no-epoch-checkpoints \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 0 \
  \
  --task speecht5 \
  --t5-task discrete_pretrain_gan \
  --label-rates 1 \
  --sample-rate 1 \
  \
  --num-workers 0 \
  --max-tokens 500 \
  --encoder-layers 2 \
  --max-speech-sample-size 100 \
  --update-freq 16 \
  --batch-ratio "[1,1]" \
  \
  --criterion speecht5 \
  --gan-loss-weight 0 \
  --gradient-penalty 0 \
  --entropy-penalty 0 \
  --entropy-threshold 1 \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-06 \
  --weight-decay 0 \
  --power 1 \
  --clip-norm 20.0 \
  --lr 2e-4 \
  --lr-scheduler polynomial_decay \
  --mask 0.3 \
  --mask-random 0.1 \
  \
  --max-update 1000000 \
  --warmup-updates 150 \
  --total-num-update 1000000 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  \
  --arch t5_transformer_base_1_layer \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --loss-weights="[10,0.1]" \
  --bart-weight 1.0 \
  --masked-lm-weight 0.0 \
  --unigram-weight 0.0 \
  --max-text-positions 100 \
  --discriminator-depth 3 \
  --discriminator-kernel 8 \
  --discriminator-causal \
  --start-gan-updates 1000000000000 \
  --policy-pretrain-weight 1 \
  --policy-loss-weight 0 \
  --init-clus-dir $INIT_CLUS_DIR \
  --use-valid-s2t \
  --post-process "none" \
  --report-similarity \
  --similarity-meanpool \
  --ce-weight 0 \
  --ctc-weight 1 \
  --report-accuracy \
  --sampling-nums 1 \
  --km-size ${topk} \
  --frame-target-classes 100 \
  --latent-vars 100 \
  --latent-groups 3 \
  --word-freq 14 \
  --word-freq-weight 500 \
  --speech-encoder-prenet-mask 0.3 \
  --speech-encoder-prenet-mask-random  0.1 \
  --speech-encoder-prenet-poisson-lambda 3.5 \
  --speech-encoder-prenet-mask-length "span-poisson" \
  --speech-encoder-prenet-replace-length -1 \
  --replace-length -1 \
  --code-loss-weight 1.0 \
  --word-count-loss-weight 500000 \
  --text-loss-ratio 1 \
  --frame-target-loss-weight 1.0 \
  --policy-pretrain-pos-weight-bce 1.5 \
  --use-codebook \
  --codebook-prob 0.3 \
  --boundary-sum-multiply 1 \
  --policy-feat-dim ${FEAT_DIM} \
  --mlm-unmasked-weight 0.5 \
  --init-model-dir $INIT_MODEL_DIR \
  --from-small-size ${small_size} \