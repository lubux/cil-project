from CNNTwitterPreprocessor import Preprocessor
import CNNTwitterPreprocessor as prep
import CNNTwitter as cn

"""
temp = Preprocessor()
temp.process("./twitter-datasets/train_pos.txt")
temp.process("./twitter-datasets/train_neg.txt")
temp.gen_data(20000,"./temp.p")
"""
#cn.train_model("./temp.p", "./twitter-datasets/train_pos.txt", "./twitter-datasets/train_neg.txt", "./", "first_test")

#cn.eval_model("./temp.p", "./", "./twitter-datasets/test_data.txt")


temp = Preprocessor()
temp.process("./twitter-datasets/train_pos_full.txt")
num_pos = temp.get_num_lines()
print("NumPos %d" % (num_pos,))
temp.process("./twitter-datasets/train_neg_full.txt")
total = temp.get_num_lines()
print("NumNeg %d" % (total - num_pos,))
print("Total %d" % (total,))
