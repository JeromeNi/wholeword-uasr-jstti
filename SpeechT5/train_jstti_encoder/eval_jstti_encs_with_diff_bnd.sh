### For all runs that uses the CNN segmenter and the quantizer within the model (i.e., JSTTI E2E-refinement)
USER_DIR=/path/to/SpeechT5/SpeechWordT5_GAN_both_discrete_l1_matching_policy_diff_bnd_encoder_only_new_version_2024/speecht5

cd $USER_DIR/../

topk=2048
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_bnd_used # be consistent with the naming of the word

# point to your data directory for a specific boundary type
DATA_DIR=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}
# point to your checkpoint directory for a specific boundary type
CKPT_PATH=${DATA_DIR}/model_codebook_0.0_clsattn_bnd_kmeans_encoder_only_retry_2024_2l_with_diff_bnd/

echo "dumping VALID_TRUE"
python dump_prediction.py $DATA_DIR --ckpt-path $CKPT_PATH/checkpoint_best.pt --user-dir $USER_DIR  --max-text-positions 100 --split "discrete_speech_valid_true_clsattnbndkmeans|text_valid_true" --postfix "valid_true_layer_0" --use-discrete-labels 0 --tgt-enc-layer 0