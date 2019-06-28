import numpy as np
from collections import defaultdict
import dill
from tqdm import tqdm
# 学習1回、隠れ層1つ、隠れ層のノード2つ

# train-nn.py
class RNN:
    def __init__(self):
        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        self.net = []
        self.feat_lab = []
        self.node = 10
        self.inp_dim = len(self.word_ids)
        self.out_dim = len(self.pos_ids)

    def prepare(self, input_file):
        data = []
        for sentence in [x for x in open(input_file)]:
            x_s = []
            y_s = []
            for one_data in sentence.rstrip().split():
                word, pos = one_data.split("_")
                word = word.lower() # 小文字で統一
                self.word_ids[word]
                self.pos_ids[pos]
                x_s.append(word)
                y_s.append(pos)
            data.append([x_s, y_s])
        for words, poses in data:
            word_vec = []
            pos_vec = []
            for word in words:
                word_vec.append(self.create_one_hot(self.word_ids[word], len(self.word_ids)))
            for pos in poses:
                pos_vec.append(self.pos_ids[pos])
            self.feat_lab.append([np.array(word_vec), np.array(pos_vec)])
        self.inp_dim = len(self.word_ids)
        self.out_dim = len(self.pos_ids)
        print(f"input_dim = {self.inp_dim}, output_dim = {self.out_dim}")

    def init_net(self):
        LOW = -0.1
        HIGH = 0.1
        w_rx = np.random.uniform(LOW, HIGH, (self.node, self.inp_dim))
        w_rh = np.random.uniform(LOW, HIGH, (self.node, self.node))
        w_oh = np.random.uniform(LOW, HIGH, (self.out_dim, self.node))
        b_r = np.random.uniform(LOW, HIGH, self.node)
        b_o = np.random.uniform(LOW, HIGH, self.out_dim)
        self.net = [w_rx, w_rh, b_r, w_oh, b_o]

    def train(self, num_iter=2):
        for i in range(num_iter):
            print(f"Epoch {i}")
            for x, y_d in self.feat_lab:
                h, p, y = self.forward_rnn(x)
                delta = self.gradient_rnn(x, h, p, y_d)
                self.update_weights(delta, 0.01)
            print(y_d, y)

    def create_one_hot(self, id_, size):
        vec = np.zeros(size)
        vec[id_] = 1
        return vec

    def softmax(self, x):
        u = np.sum(np.exp(x))
        return np.exp(x)/u

    def forward_rnn(self, x):
        w_rx, w_rh, b_r, w_oh, b_o = self.net
        h = [0] * len(x); p = [0] * len(x); y = [0] * len(x)
        for t in range(len(x)):
            if t > 0:
                h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t - 1]) + b_r)
            else:
                h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)
            y[t] = np.argmax(p[t])
        return np.array(h), np.array(p), np.array(y)

    def gradient_rnn(self, x, h, p, y_d):
        w_rx, w_rh, b_r, w_oh, b_o = self.net
        d_w_rx = np.zeros(w_rx.shape)
        d_w_rh = np.zeros(w_rh.shape)
        d_w_oh = np.zeros(w_oh.shape)
        d_b_r = np.zeros(b_r.shape)
        d_b_o = np.zeros(b_o.shape)
        d_r_d = np.zeros(len(b_r))

        for t in range(len(x))[::-1]:
            p_d = self.create_one_hot(y_d[t], self.out_dim)
            d_o_d = p_d - p[t]
            d_w_oh += np.outer(d_o_d, h[t])
            d_b_o += d_o_d
            d_r = np.dot(d_r_d, w_rh) + np.dot(d_o_d, w_oh)
            d_r_d = d_r * (1 - h[t] ** 2)
            d_w_rx += np.outer(d_r_d, x[t])
            d_b_r += d_r_d
            if t != 0:
                d_w_rh += np.outer(d_r_d, h[t-1])
        return d_w_rx, d_w_rh, d_b_r, d_w_oh, d_b_o

    def update_weights(self, delta, lam = 0.1):
        for i in range(5):
            self.net[i] += lam * delta[i]

    def test(self, sentence):
        words = sentence.rstrip().split()
        vec = []
        for word in words:
            if word in self.word_ids.keys():
                vec.append(self.create_one_hot(self.word_ids[word], len(self.word_ids)))
            else:
                vec.append(np.zeros(len(self.word_ids)))
        vec = np.array(vec)
        h_, p_, y = self.forward_rnn(vec)
        res = []
        id2pos = {self.pos_ids[k]: k for k in self.pos_ids}
        for pred in y:
            res.append(str(id2pos[pred]))
        return " ".join(res)

if __name__ == "__main__":
    input_path = "../../data/wiki-en-train.norm_pos"
    test_path = "../../data/wiki-en-test.norm"
    output_path = "ans"

    net = RNN()
    print("Loading data...")
    net.prepare(input_path)
    net.init_net()
    print("Training...")
    net.train(3)
    print("Predicting...")
    ans = []
    for sentence in open(test_path):
        pred = net.test(sentence)
        ans.append(pred)
    with open(output_path, mode="w") as fw:
        fw.write("\n".join(ans))
    print("Finished!")

# $ ../../script/gradepos.pl ../../data/wiki-en-test.pos ans
## result
"""
Accuracy: 73.46 % (3352/4563)

Most common mistakes:
JJ - -> NN       172
NNS - -> NN      157
NNP - -> NN      81
RB - -> NN       79
VBG - -> NN      48
-RRB - - -> NN    47
-LRB - - -> NN    46
DT - -> NN       45
VBN - -> NN      41
VBN - -> JJ      38
"""

