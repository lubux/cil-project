import CNNTwitterPreprocessor as prep
"""
NumPos 1127644
NumNeg 1142838
Total 2270482
"""
train = 600000
val = 200000
test = 200000

prep.create_big_files(train, val, test,
                      "./twitter-datasets/train_pos_full.txt",
                      "./twitter-datasets/train_pos_huge.txt",
                      "./twitter-datasets/val_pos_huge.txt",
                      "./twitter-datasets/test_pos_huge.txt")

prep.create_big_files(train, val, test,
                      "./twitter-datasets/train_neg_full.txt",
                      "./twitter-datasets/train_neg_huge.txt",
                      "./twitter-datasets/val_neg_huge.txt",
                      "./twitter-datasets/test_neg_huge.txt")
