import CNNTwitter as cn
import argparse
test_data_p = "./twitter-datasets/val_pos_large.txt"
test_data_n = "./twitter-datasets/val_neg_large.txt"
store_vocab = "./model/model4/temp.p"
store_model = "./model/model4/"

parser = argparse.ArgumentParser(description='Evaluate model on testset')
parser.add_argument('-m', '--modelname', type=str, nargs=1,
                    help='the name of the model in ./models', default="model4")
parser.add_argument('-tp', '--test_pos', type=str, nargs=1,
                    help='the positive tweets from the testset', default=test_data_p)
parser.add_argument('-tn', '--test_neg', type=str, nargs=1,
                    help='the negative tweets from the testset', default=test_data_n)
args = parser.parse_args()

if args.modelname is not None:
    store_model = "./model/" + args.modelname + "/"

loss, acc = cn.evaluate_on_test(store_vocab, args.test_pos, args.test_neg, store_model)
print("Loss on test set: %.4f, Accuracy on test set: %.2f%%" % (loss, acc * 100))
