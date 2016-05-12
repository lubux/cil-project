import cnntwitter.CNNTwitter as cn
test_data = "./twitter-datasets/test_data.txt"
store_vocab = "./model/temp.p"
store_model = "./model/"
cn.eval_kaggle_test(store_vocab, store_model, test_data)