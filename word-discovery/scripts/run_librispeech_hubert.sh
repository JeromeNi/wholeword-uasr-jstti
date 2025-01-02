# bash run_librispeech_hubert.sh vg-hubert_3 9 4096 0.7 max clsAttn 1 


model=$1 # vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
k=$3 # 4096
threshold=$4 # 0.7
reduce_method=$5 # mean, max etc.
segment_method=$6 # clsAttn
data_root=$7
data_json=$8
dataset=$9

model_root=/path/to/uncompressed/vghubert/tar
seed=1
save_root=/path/to/save_data/${model}_${tgt_layer_for_attn}_${k}_${threshold}_${reduce_method}_${segment_method}_${seed}_${dataset} # save intermediate data

mkdir -p ${save_root}
python ../save_seg_feats_libri.py \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/${model} \
--audio_base_path ${data_root} \
--save_root ${save_root} \
--data_json ${data_json} \
--dataset ${dataset} \
# --ori_feats
