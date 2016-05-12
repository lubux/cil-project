import time
import tensorflow as tf
import numpy as np
import pickle
from cnntwitter.CNNTwitterPreprocessor import Preprocessor
import os
import re

class CNNTwitterModel:
    def __init__(self, is_train, params):
        self._create_graph(is_train, params)

    def _create_graph(self, is_train, params):
        self.sentence_length = params.sentence_length
        self.batch_size = params.batch_size
        self.embedding_size = params.embedding_size
        self.vocab_size = params.vocab_size
        self.filter_num = params.filter_num
        self.filter_size = params.filter_size
        self.dropout_prob = params.dropout_prob
        self.learning_rate = params.learning_rate
        self.l2_reg_lambda = params.l2_reg_lambda

        # input_data in form of batch_size*(sentence)
        self.input_data = tf.placeholder(tf.int32,
                                         [self.batch_size, self.sentence_length],
                                         name="input_data")
        # the corresponding binary classification
        self.targets = tf.placeholder(tf.float32, [self.batch_size, 2], name="targets")

        with tf.device("/cpu:0"):
            # embed words
            word_embedding = tf.get_variable("word_embedding", [self.vocab_size, self.embedding_size])
            inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)
            # add channel 1 for conv layer
            inputs_channel = tf.expand_dims(inputs, -1)

        # convolution (
        # input_shape: [batch, in_height, in_width, in_channels],
        # filter_shape: [filter_height, filter_width, in_channels, out_channels])

        # the filters to learn
        filters = tf.get_variable("filters", [self.filter_size, self.embedding_size, 1, self.filter_num])
        # biases that are added to the convolution result
        bias = tf.get_variable("biases", [self.filter_num])
        # perform convolution
        conv = tf.nn.conv2d(inputs_channel,
                            filters,
                            [1, 1, 1, 1],
                            padding='VALID',
                            name="convolution")
        # add bias to result
        added_bias = tf.nn.bias_add(conv, bias)
        # apply relu activation function
        res = tf.nn.relu(added_bias, name="relu_func")
        # perform max_pooling
        pooled_res = tf.nn.max_pool(res,
                                    ksize=[1, self.sentence_length - self.filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="max_pool")
        pooled_res = tf.reshape(pooled_res, [-1, self.filter_num])

        if is_train and self.dropout_prob < 1:
            pooled_res = tf.nn.dropout(pooled_res, self.dropout_prob)

        # add l2 regulation
        l2_reg = tf.constant(0.0)
        # soft max
        softmax_w = tf.get_variable("softmax_w", [self.filter_num, 2])
        softmax_b = tf.get_variable("softmax_b", [2])
        self.logits = tf.nn.xw_plus_b(pooled_res, softmax_w, softmax_b)
        losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.targets)
        l2_reg += tf.nn.l2_loss(softmax_w)
        l2_reg += tf.nn.l2_loss(softmax_b)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_reg
        self.probabilities = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1, name="predictions")

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if not is_train:
            return
        # only in training mode
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)


class CNNTwitterParams:
    def __init__(self):
        self.sentence_length = 64
        self.batch_size = 20
        self.embedding_size = 100
        self.vocab_size = 20000
        self.filter_num = 300
        self.filter_size = 5
        self.dropout_prob = 0.5
        self.learning_rate = 1e-4
        self.max_max_epoch = 39
        self.init_scale = 0.05
        self.l2_reg_lambda = 1e-4


def _run_epoch(session, model, data, train_op, verbose=True):
    def run_step(in_x, in_y):
        feed_dict = {
          model.input_data: in_x,
          model.targets: in_y,
        }
        _, it, loss_step, accuracy = session.run([train_op, model.global_step, model.loss, model.accuracy], feed_dict)
        return it, loss_step, accuracy
    global_step = 0
    loss = 0
    acc = 0
    count = 0
    for step, (x, y) in enumerate(mat_iterator(data, model.batch_size)):
        global_step, loss_it, acc_it = run_step(x, y)
        loss += loss_it
        acc += acc_it
        count += 1
        if verbose and global_step % 1000 == 0:
            print("Step: %d, loss: %f accuracy %f" % (global_step, loss / count, acc / count))
    return global_step, loss / count, acc / count


def train_model(pre_data_path,
                data_path_pos, data_path_neg,
                data_eval_pos, data_eval_neg,
                model_store_dir, model_name,
                config=CNNTwitterParams(), eval_config=CNNTwitterParams(),
                out_log="./log"):

    [max_sent_len, word_to_id, vocab] = load_data(pre_data_path)
    data_train_pos = get_data_mat(data_path_pos, max_sent_len, word_to_id, True)
    data_train_neg = get_data_mat(data_path_neg, max_sent_len, word_to_id, False)
    data_train = merge_and_shuffle(data_train_pos, data_train_neg)

    data_eval_p = get_data_mat(data_eval_pos, max_sent_len, word_to_id, True)
    data_eval_n = get_data_mat(data_eval_neg, max_sent_len, word_to_id, False)
    data_eval = merge_and_shuffle(data_eval_p, data_eval_n)

    train_time = 0
    config.vocab_size = len(vocab)
    config.sentence_length = max_sent_len
    eval_config.vocab_size = len(vocab)
    eval_config.sentence_length = max_sent_len
    eval_config.dropout_prob = 1.0

    print("Start Training")
    with tf.Graph().as_default(), tf.Session() as session, open(out_log, "w") as log:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = CNNTwitterModel(True, config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_eval = CNNTwitterModel(False, eval_config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for i in range(0, config.max_max_epoch):
            print("Epoch: %d started" % (i + 1))
            start_time = time.time()
            _, loss, acc =_run_epoch(session, m, data_train, m.train_op, verbose=True)
            end_time = time.time()
            train_time += (end_time-start_time)
            save_path = saver.save(session, os.path.join(model_store_dir, model_name + ".ckpt"), global_step=(i + 1))
            print("Training Summary -> Loss: %f, Accuracy %f" % (loss, acc))
            print("Model saved in file: %s" % save_path)

            # shuffle data
            np.random.shuffle(data_train)

            print("Start evaluation for Epoch %d" % (i+1))
            _, loss, acc = _run_epoch(session, m_eval, data_eval, tf.no_op(), verbose=False)
            print("Evaluation -> Loss: %f, Accuracy %f" % (loss, acc))
            log.write("Epoch %d Loss: %f, Accuracy %f\n" % (i+1, loss, acc))
            log.flush()

    print("Training Finished after: %f seconds" % train_time)


def eval_kaggle_test(pre_data_path, model_path, path_testdata, config=CNNTwitterParams(), outfile="./out.csv"):
    [max_sent_len, word_to_id, vocab] = load_data(pre_data_path)
    config.vocab_size = len(vocab)
    config.sentence_length = max_sent_len
    config.dropout_prob = 1.0
    data = get_data_test(path_testdata, max_sent_len, word_to_id)

    with tf.Graph().as_default():
        session = tf.Session()
        with session.as_default(), open(outfile, "w") as out:
            with tf.variable_scope("model", reuse=False):
                model = CNNTwitterModel(False, config)

            tf.initialize_all_variables().run()
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)

            count = 1
            out.write("Id,Prediction\n")
            print("Id,Prediction")
            for step, (x, y) in enumerate(mat_iterator(data, model.batch_size)):
                feed_dict = {
                    model.input_data: x
                }
                prediction = session.run([model.predictions], feed_dict)
                for it in np.nditer(prediction):
                    if it == 0:
                        # idx 0 is positive
                        a = 1
                    else:
                        # idx 1 is negative
                        a = -1
                    out.write("%d,%d\n" % (count, a))
                    print("%d,%d" % (count, a))
                    count += 1


def merge_and_shuffle(pos_data, neg_data):
    data = np.concatenate((pos_data, neg_data), axis=0)
    np.random.shuffle(data)
    return data


def get_data_mat(path, max_sent_len, vocab_to_id, is_pos):
    id_unk = vocab_to_id[Preprocessor.TOKEN_UNKOWN]
    id_pad = vocab_to_id[Preprocessor.TOKEN_PAD]
    res = []
    with open(path, "r") as f:
        for step, line in enumerate(f):
            tokens = line.split()
            ids = [vocab_to_id.get(x, id_unk) for x in tokens]
            line_id = np.empty(max_sent_len + 2)
            line_id.fill(id_pad)
            line_id[:len(ids)] = np.asarray(ids)
            if is_pos:
                line_id[max_sent_len:] = np.asarray([1, 0])
            else:
                line_id[max_sent_len:] = np.asarray([0, 1])
            res.append(line_id)
    return np.asarray(res)


def get_data_test(path, max_sent_len, vocab_to_id):
    id_unk = vocab_to_id[Preprocessor.TOKEN_UNKOWN]
    id_pad = vocab_to_id[Preprocessor.TOKEN_PAD]
    res = []
    with open(path, "r") as f:
        for step, line in enumerate(f):
            line = re.sub(r'^\d+,', "", line, count=1)
            tokens = line.split()
            ids = [vocab_to_id.get(x, id_unk) for x in tokens]
            line_id = np.empty(max_sent_len + 2)
            line_id.fill(id_pad)
            line_id[:len(ids)] = np.asarray(ids)
            res.append(line_id)
    return np.asarray(res)


def mat_iterator(data, batch_size):
    (num_lines, sen_len) = data.shape
    num_iters = num_lines // batch_size
    from_it = 0
    to_it = batch_size
    for i in range(num_iters):
        y = data[from_it:to_it, (sen_len-2):]
        x = data[from_it:to_it, :(sen_len-2)]
        from_it += batch_size
        to_it += batch_size
        yield (x, y)


def load_data(file_path):
    return pickle.load(open(file_path, "rb"))
