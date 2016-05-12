import pickle


class Preprocessor:

    TOKEN_PAD = "<p>"
    TOKEN_UNKOWN = "<u>"

    def __init__(self):
        self.max_sent_len = 0
        self.vocab_to_count = {}

    def process(self, path):
        with open(path, "r") as f:
            for line in f:
                tokens = line.split()
                if len(tokens) > self.max_sent_len:
                    self.max_sent_len = len(tokens)
                for token in tokens:
                    if token in self.vocab_to_count:
                        count = self.vocab_to_count[token]
                        count += 1
                        self.vocab_to_count[token] = count
                    else:
                        self.vocab_to_count[token] = 1

    def get_vocab_size(self):
        return len(self.vocab_to_count)

    def get_max_sent_len(self):
        return self.max_sent_len

    def get_word_to_id(self, max_vocab_size):
        max_vocab = max_vocab_size - 2
        items = sorted(self.vocab_to_count.items(), key=lambda t: t[1], reverse=True)
        vocab = [x[0] for x in items[:max_vocab]]
        vocab.append(self.TOKEN_PAD)
        vocab.append(self.TOKEN_UNKOWN)
        vocab.sort()
        return vocab, dict(zip(vocab, range(len(vocab))))

    def gen_data(self, max_vocab_size, out_path="./out.p"):
        vocab, word_to_id = self.get_word_to_id(max_vocab_size)
        pickle.dump([self.max_sent_len, word_to_id, vocab], open(out_path, "wb"))


def _split_file(file_pos, num_skip, num_entries, out):
    with open(file_pos, "r") as r_f, open(out, "w") as w_f:
        cur_num = 0
        for step, line in enumerate(r_f):
            if step >= num_skip:
                if cur_num == num_entries:
                    break
                w_f.write(line)
                cur_num += 1


def split_to_eval_set(file_pos, file_neg, num_skip, num_entries, out_pos="./eval_pos.txt", out_neg="./eval_neg.txt"):
    _split_file(file_pos, num_skip, num_entries, out_pos)
    _split_file(file_neg, num_skip, num_entries, out_neg)
