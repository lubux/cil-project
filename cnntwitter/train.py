from cnntwitter.CNNTwitterPreprocessor import Preprocessor
import cnntwitter.CNNTwitter as cn

train_pos = "./twitter-datasets/train_pos.txt"
train_neg = "./twitter-datasets/train_neg.txt"
store_vocab = "./model/temp.p"
store_model = "./model/"
eval_pos = "./twitter-datasets/eval_pos.txt"
eval_neg = "./twitter-datasets/eval_neg.txt"
model_name = "twitter_train"
vocab_size = 20000

temp = Preprocessor()
temp.process(train_pos)
temp.process(train_neg)
temp.gen_data(vocab_size, store_vocab)

cn.train_model(store_vocab,
               train_pos, train_neg,
               eval_pos, eval_neg,
               store_model, model_name)
