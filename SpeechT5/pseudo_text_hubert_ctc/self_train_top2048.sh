# The following script is originally designed to run on a slurm-based server,
# configured as a 2-node setup, where each node has 4 GPUs. You will need to change
# the script to adapt to your distributed training environment!
# Some modifications in the yaml may be needed for your distributed training environment
# You may need to adjust the UPDATE_FREQ so that DISTRIBUTED_WORLD_SIZE x UPDATE_FREQ is at least as large as this setup

DATA_PATH=/path/to/self_train_tsvs

CONFIG_DIR=config
DISTRIBUTED_WORLD_SIZE=8
DISTRIBUTED_PORT=12345
UPDATE_FREQ='[4]'
TRAIN_SUBSET='train'
VALID_SUBSET='valid'

# where to save the model
SAVE_DIR=${DATA_PATH}/model_hubert_large_valid_on_valid
CONFIG_NAME=base_hubert

# the HuBERT-large model with no fine-tuning
RESTORE_FILE=/path/to/hubert_large_ll60k.pt



srun fairseq-hydra-train task.data=$DATA_PATH \
                    task.label_dir=$DATA_PATH \
                    model.w2v_path=$RESTORE_FILE \
                    checkpoint.save_dir=$SAVE_DIR \
                    distributed_training.distributed_world_size=$DISTRIBUTED_WORLD_SIZE \
                    optimization.update_freq=$UPDATE_FREQ \
                    dataset.train_subset=$TRAIN_SUBSET \
                    dataset.valid_subset=$VALID_SUBSET \
                    distributed_training.distributed_port=${DISTRIBUTED_PORT} \
                    --config-dir $CONFIG_DIR \
                    --config-name $CONFIG_NAME
