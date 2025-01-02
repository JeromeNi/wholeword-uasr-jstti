#!/bin/bash

set -e

topk=2048
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_any_word_seg_bnds # be consistent with the naming of the word boundaries
# point to the directory structure in step 8 + /seg_feats
feat_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats

lab_dir=$feat_dir/km_idx_clsattnbndkmeans
wrd_bnd_type_for_vghubert_bnds=your_flag_for_vghubert_bnds
# note that km_dir points to the km model obtained from VG-HuBERT boundaries
km_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type_for_vghubert_bnds}/top_${topk}/km_dir_${topk}/seg_feats/km_model

nshard=1

mkdir -p -m 777 $lab_dir


python dump_km_label_faiss.py $feat_dir "train" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "valid" "${km_dir}" 1 0 $lab_dir
python dump_km_label_faiss.py $feat_dir "valid_true" "${km_dir}" 1 0 $lab_dir


for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/train_${rank}_${nshard}.km
done > $lab_dir/train.km

mv $lab_dir/valid_0_1.km $lab_dir/valid.km
mv $lab_dir/valid_true_0_1.km $lab_dir/valid_true.km

for rank in $(seq 0 $((nshard - 1))); do
  rm $lab_dir/train_${rank}_${nshard}.km
done

