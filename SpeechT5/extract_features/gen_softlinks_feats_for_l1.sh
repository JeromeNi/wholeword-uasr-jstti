topk=2048
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=yyy # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
exp_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/


mkdir -p $exp_dir/feats_for_l1_clsattnbndkmeans

cd $exp_dir/feats_for_l1_clsattnbndkmeans

which_set="valid_true"

ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/file_list.txt ${which_set}.files
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/data.km ${which_set}.km
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/data.npy ${which_set}frame.npy
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/data.lengths ${which_set}frame.lengths
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/data_dict_bnd.pkl ${which_set}boundary.pkl
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/data_frame_clus.km ${which_set}_frame_clus.km
ln -s ../discrete_speech_${which_set}_clsattnbndkmeans/bnd_errors.txt ${which_set}_bnd_errors.txt
ln -s ../../${which_set}_words.txt ${which_set}.wrd
