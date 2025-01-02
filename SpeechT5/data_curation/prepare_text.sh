topk=2048
text_path=/path/to/top_${topk}
target_dir=/path/to/top_${topk}/km_dir_${topk}

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/train_words.txt --only-source --destdir $target_dir/text_train --srcdict $text_path/dict.txt
mv $target_dir/text_train/train.bin $target_dir/text_train/text_train.bin
mv $target_dir/text_train/train.idx $target_dir/text_train/text_train.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/valid_words.txt --only-source --destdir $target_dir/text_valid --srcdict $text_path/dict.txt
mv $target_dir/text_valid/train.bin $target_dir/text_valid/text_valid.bin
mv $target_dir/text_valid/train.idx $target_dir/text_valid/text_valid.idx

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $text_path/valid_true_words.txt --only-source --destdir $target_dir/text_valid_true --srcdict $text_path/dict.txt
mv $target_dir/text_valid_true/train.bin $target_dir/text_valid_true/text_valid_true.bin
mv $target_dir/text_valid_true/train.idx $target_dir/text_valid_true/text_valid_true.idx

mkdir -p $target_dir/text_for_l1
cp $target_dir/text_train/text_train.bin $target_dir/text_for_l1/train.bin
cp $target_dir/text_train/text_train.idx $target_dir/text_for_l1/train.idx

# assuming you have KenLM installed
lmplz -o 4 < $text_path/train_words.txt --discount_fallback --prune 0 0 0 3 > "$target_dir/kenlm.wrd.o40003.arpa"
build_binary "$target_dir/kenlm.wrd.o40003.arpa" "$target_dir/kenlm.wrd.o40003.bin"
