
N=4 # this is the number of GPUs

which_set="valid" # which set to use (train, valid, valid_true)
flag="prom_xx_height_xx" # useful for recording the current hyperparameters in infere.py
VAD_FILE=/path/to/vads/${which_set}.vad # point to the output directory of `get_vad_simple.py`
OUTPUT_DIR=/path/to/saved_boundary_outputs/${which_set}_${flag}/ # where to save the boundary predictions
SAVE_DIR=/path/to/trained_models # where the trained model is saved, i.e., the path in the yaml file for `save_dir`

mkdir -p $OUTPUT_DIR

python infere.py $N ${VAD_FILE} ${SAVE_DIR}/checkpoint_best.pt ${OUTPUT_DIR}/

cat $OUTPUT_DIR/* > $OUTPUT_DIR/all_word_boundaries.txt