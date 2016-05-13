import cnntwitter.CNNTwitterPreprocessor as prep

train_size_per_class = 250000
valid_per_class = 50000

prep.split_to_eval_set(
    "./twitter-datasets/train_pos_full.txt",
    "./twitter-datasets/train_neg_full.txt",
    0, train_size_per_class,
    out_pos="./twitter-datasets/train_pos_large.txt",
    out_neg="./twitter-datasets/train_neg_large.txt")

prep.split_to_eval_set(
    "./twitter-datasets/train_pos_full.txt",
    "./twitter-datasets/train_neg_full.txt",
    train_size_per_class + 50000, valid_per_class,
    out_pos="./twitter-datasets/val_pos_large.txt",
    out_neg="./twitter-datasets/val_neg_large.txt")