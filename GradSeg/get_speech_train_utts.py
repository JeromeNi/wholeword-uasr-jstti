# This script is used to get a new tsv file for each pruned dataset for the 100 utterances used to train the GradSeg model.
# Modify `gradseg_training_fn` with the path to your train_gradseg_1024_100utts.tsv file. Modify `read_tsv` to the train.tsv you prepared for the topk pruned dataset, and modify `write_tsv` to the new tsv file with 100 utterances, in the same directory as `train.tsv`.

import os

topk=4096
gradseg_training_fn = '/path/to/train_gradseg_1024_100utts.tsv'

read_tsv = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/tsvs/train.tsv'
write_tsv = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/tsvs/train_100utts.tsv'

with open(read_tsv, 'r') as f:
    lines = f.readlines()
    
parent_dir = lines[0]
all_files = lines[1:]

file_dict = {}

for file in all_files:
    file_dict[file.strip().split('\t')[0]] = file.strip().split('\t')[1]
    
with open(gradseg_training_fn, 'r') as f:
    training_lines = f.readlines()[1:]
    
with open(write_tsv, 'w') as fw:
    fw.write(parent_dir)
    for training_line in training_lines:
        file = training_line.strip().split('\t')[0]
        fw.write(file + '\t' + file_dict[file] + '\n')
    