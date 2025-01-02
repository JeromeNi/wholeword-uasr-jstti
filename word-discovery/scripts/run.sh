topk=1024 # or 2048, or 4096
model_card=vg-hubert_3
curated_audio_path=/path/to/LibriSpeech_clean_pruned_top${topk}_no_sil

bash run_librispeech_hubert.sh ${model_card} 9 4096 0.7 mean clsAttnBnd ${curated_audio_path} ${curated_audio_path}/ls_train_clean_pruned_top${topk}_coco_style_no_sil_train.json librispeech_clean_pruned_top${topk}_no_sil_train_set

bash run_librispeech_hubert.sh ${model_card} 9 4096 0.7 mean clsAttnBnd ${curated_audio_path} ${curated_audio_path}/ls_train_clean_pruned_top${topk}_coco_style_no_sil_valid.json librispeech_clean_pruned_top${topk}_no_sil_valid_set

bash run_librispeech_hubert.sh ${model_card} 9 4096 0.7 mean clsAttnBnd ${curated_audio_path} ${curated_audio_path}/ls_train_clean_pruned_top${topk}_coco_style_no_sil_valid_true.json librispeech_clean_pruned_toptop${topk}_no_sil_valid_true_set