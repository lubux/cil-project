import CNNTwitter as cn
import argparse
test_data = "./twitter-datasets/test_data.txt"

parser = argparse.ArgumentParser(description='Kaggle tweet classification')
parser.add_argument('-m', '--modelname', type=str, nargs=1,
                    help='the name of the model in ./models', default="model5")
parser.add_argument('-t', '--test_data', type=str, nargs=1,
                    help='the testdata for the kaggle submission', default=test_data)

args = parser.parse_args()
store_model = "./model/" + args.modelname + "/"
store_vocab = "./model/" + args.modelname + "/temp.p"

cn.eval_kaggle_test(store_vocab, store_model, args.test_data)
