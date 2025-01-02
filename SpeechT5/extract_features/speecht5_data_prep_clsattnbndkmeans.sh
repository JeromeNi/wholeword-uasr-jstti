topk=2048
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=yyy # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
exp_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/

cd $exp_dir

which_set="valid_true"

mkdir -p discrete_speech_${which_set}_clsattnbndkmeans

# Modify the following paths to match your setup
topk=1024
WORD_SEG_DIR=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil/word_segmentation_containing_*_utts.pkl/
FILE_LIST_DIR=/path/to/file_lists
WORD_TOKEN_KM_DIR=${exp_dir}/seg_feats/km_idx_clsattnbndkmeans
FRAME_LEVEL_ACOUSTIC_TARGET_DIR=/path/to/hubert_feats/label_hubert_base_L6
AUDIO_DIR=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil/
FRAME_LEVEL_FEATS_DIR=/path/to/hubert_feats/feat_hubert_large_L21

ln -s $FILE_LIST_DIR/${which_set}_bnd_errors.txt discrete_speech_${which_set}_clsattnbndkmeans/bnd_errors.txt

ln -s $WORD_TOKEN_KM_DIR/${which_set}.km discrete_speech_${which_set}_clsattnbndkmeans/data.km

ln -s $WORD_SEG_DIR/${which_set}_utts.pkl discrete_speech_${which_set}_clsattnbndkmeans/data_dict_bnd.pkl

ln -s $FRAME_LEVEL_ACOUSTIC_TARGET_DIR/${which_set}.km discrete_speech_${which_set}_clsattnbndkmeans/data_frame_clus.km

ln -s $FILE_LIST_DIR/file_list.txt discrete_speech_${which_set}_clsattnbndkmeans/file_list.txt

ln -s $AUDIO_DIR/ls_train_clean_pruned_top${topk}_gradseg_style_no_sil_${which_set}.json discrete_speech_${which_set}_clsattnbndkmeans/meta_data.json


ln -s $FRAME_LEVEL_FEATS_DIR/${which_set}_0_1.npy discrete_speech_${which_set}_clsattnbndkmeans/data.npy

ln -s $FRAME_LEVEL_FEATS_DIR/${which_set}_0_1.len discrete_speech_${which_set}_clsattnbndkmeans/data.lengths


