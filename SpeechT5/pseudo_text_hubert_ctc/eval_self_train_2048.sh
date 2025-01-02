FAIRSEQ_ROOT=/path/to/fairseq

DATA_DIR=/path/to/self_train_tsvs
CKPT_PATH=/path/to/self_train_tsvs/model_hubert_large_valid_on_valid/checkpoint_best.pt


cd $FAIRSEQ_ROOT

python examples/speech_recognition/new/infer.py \
  --config-dir config/decode \
  --config-name infer_viterbi \
  task.data=$DATA_DIR \
  task.normalize=false \
  task.labels=["wrd"] \
  decoding.results_path=$DATA_DIR/model_hubert_large_greedy_decoding_log_on_valid_wer \
  common_eval.results_path=$DATA_DIR/model_hubert_large_greedy_decoding_log_on_valid_wer \
  common_eval.path=$CKPT_PATH \
  dataset.gen_subset=valid_true_gt \