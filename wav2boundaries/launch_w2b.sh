config_file='config/librispeech-clean-no-sil-topk-w2b.yaml'
fairseq_dir='/path/to/fairseq_fork/'

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PWD

root_home=$PWD

dataset=$(cat $config_file | grep data: | head -n 1 | cut -d ' ' -f 4)
config_dir=$root_home/$(dirname $config_file)
config_name=$(echo $(basename $config_file) | cut -d '.' -f 1)

PYTHONPATH=$PYTHONPATH:$fairseq_dir python $fairseq_dir/fairseq_cli/hydra_train.py -m --config-dir $config_dir --config-name $config_name task.data=${dataset}
