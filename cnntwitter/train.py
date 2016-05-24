from CNNTwitterPreprocessor import Preprocessor
import CNNTwitter as cn
import CNNTwitterPreprocessor as prep
import os.path
import argparse

train_pos_full = "./twitter-datasets/train_pos_full.txt"
train_neg_full = "./twitter-datasets/train_neg_full.txt"

parser = argparse.ArgumentParser(description='Evaluate model on testset')
parser.add_argument('-s', '--train_size', type=str, nargs=1,
                    help='indicates which training files to use (small, large, huge) default huge', default="huge")
parser.add_argument('-mn', '--modelname', type=str, nargs=1,
                    help='the name of the model', default="twitter_train")
args = parser.parse_args()

if args.train_size is 'huge':
    train_pos = "./twitter-datasets/train_pos_huge.txt"
    train_neg = "./twitter-datasets/train_neg_huge.txt"
    store_vocab = "./model/temp.p"
    store_model = "./model/"
    val_pos = "./twitter-datasets/val_pos_huge.txt"
    val_neg = "./twitter-datasets/val_neg_huge.txt"
    model_name = args.modelname
    vocab_size = 30000
elif args.train_size is 'large':
    train_pos = "./twitter-datasets/train_pos_large.txt"
    train_neg = "./twitter-datasets/train_neg_large.txt"
    store_vocab = "./model/temp.p"
    store_model = "./model/"
    val_pos = "./twitter-datasets/val_pos_large.txt"
    val_neg = "./twitter-datasets/val_neg_large.txt"
    model_name = args.modelname
    vocab_size = 30000
else:
    train_pos = "./twitter-datasets/train_pos.txt"
    train_neg = "./twitter-datasets/train_neg.txt"
    store_vocab = "./model/temp.p"
    store_model = "./model/"
    val_pos = "./twitter-datasets/val_pos.txt"
    val_neg = "./twitter-datasets/val_neg.txt"
    model_name = args.modelname
    vocab_size = 30000

#if not os.path.isfile(val_pos):
#    prep.split_to_eval_set(train_pos_full, train_neg_full, 500000, 10000, out_pos=val_pos, out_neg=val_neg)

temp = Preprocessor()
print("Preprocess positive training set")
temp.process(train_pos)
print("Preprocess negative training set")
temp.process(train_neg)
temp.gen_data(vocab_size, store_vocab)
print("Vocab Size: %s MaxSentenceLength: %d" % (temp.get_vocab_size(), temp.get_max_sent_len()))

cn.train_model(store_vocab,
               train_pos, train_neg,
               val_pos, val_neg,
               store_model, model_name)

