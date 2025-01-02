#!/bin/bash

set -e

topk=2048
feature_type=xxx # be consistent with the naming of the features
wrd_bnd_type=your_flag_for_vghubert_bnds # be consistent with the naming of the word boundaries
# point to the directory structure in step 8
feat_dir=/path/to/data/libri-train-clean-top-${topk}/feat_${feature_type}_no_sil_${wrd_bnd_type}/top_${topk}/km_dir_${topk}/seg_feats
km_dir=$feats_dir/km_model
nclus=${topk}

nshard=1

mkdir -p -m 777 $(dirname $km_dir)
python learn_kmeans_faiss.py $feat_dir "train" ${nshard} "${km_dir}" ${nclus} --percent -1
