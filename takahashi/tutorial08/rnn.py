import numpy as np
from collections import defaultdict
from tqdm import tqdm


class trainRNN:
    def __init__(self, epoch: int, node: int):
        # hyper parameter
        self.epoch = epoch
        self.node = node

        # 単語 ID と 品詞 ID
        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        # 単語 ID のサイズ, 品詞 ID のサイズ (0 で初期化)
        self.vocab_size = 0
        self.pos_size = 0
        # 文ごとの単語と品詞の onehot ベクトルのリスト
        self.feat_lab = []
        # 重みとバイアスをパラメータとしたリスト
        self.net = []
        # 伝播する誤差のリスト
        self.err = []

    def create_onehot(self, id, size):
        v = np.zeros(size)
        v[id] = 1
        return v

    def create_feature(self, input_file: str):
        # word_ids (単語 ID) と pos_ids (品詞 ID) の作成
        for line in open(input_file, "r", encoding="utf-8"):
            for word_pos in line.strip().split():
                word, pos = word_pos.split("_")
                self.word_ids[word]
                self.pos_ids[pos]

        self.vocab_size = len(self.word_ids)
        self.pos_size = len(self.pos_ids)

        for line in open(input_file, "r", encoding="utf-8"):
            word_pos_list = line.strip().split()
            # 文ごとに単語と品詞の onehot ベクトルをリストに加える
            ws = [None for _ in range(len(word_pos_list))]
            ps = [None for _ in range(len(word_pos_list))]
            for i, word_pos in enumerate(word_pos_list):
                word, pos = word_pos.split("_")
                ws[i] = self.create_onehot(self.word_ids[word], self.vocab_size)
                ps[i] = self.create_onehot(self.pos_ids[pos], self.pos_size)
            self.feat_lab.append((ws, ps))

    def init_network(self):
        rn = np.random.rand
        # 入力に対する重み
        w_rx = rn(self.node, self.vocab_size) / 5 - 0.1
        # 次の時刻に伝播する重み
        w_rh = rn(self.node, self.node) / 5 - 0.1
        # 出力に対する重み
        w_oh = rn(self.pos_size, self.node) / 5 - 0.1

        # バイアス
        b_r = rn(self.node) / 5 - 0.1
        b_o = rn(self.pos_size) / 5 - 0.1

        self.net = [w_rx, w_rh, w_oh, b_r, b_o]

    def softmax(self, vec):
        const = max(vec)
        exp_vec = np.exp(vec - const)
        sum_vec = np.sum(exp_vec)
        return exp_vec / sum_vec

    def init_error(self):
        # 入力層, 隠れ層の重み勾配
        dw_rx = np.zeros((self.node, self.vocab_size))
        dw_rh = np.zeros((self.node, self.node))
        db_r = np.zeros(self.node)

        # 出力層の重み勾配
        dw_oh = np.zeros((self.pos_size, self.node))
        db_o = np.zeros(self.pos_size)

        self.err = [dw_rx, dw_rh, db_r, dw_oh, db_o]

    def forward(self, words):
        # words : 文の onehot ベクトル
        # 単語数
        vocab = len(words)
        # 時刻 t における各値
        h = [np.ndarray for _ in range(vocab)]  # 隠れ層の値 h[t]
        p = [np.ndarray for _ in range(vocab)]  # 出力の確率分布 p[t]
        y = [0 for _ in range(vocab)]

        w_rx, w_rh, w_oh, b_r, b_o = self.net
        for t in range(vocab):
            if t != 0:
                # 入力,前の時刻の両方の重みで現時刻の隠れ層の値を求める
                current = np.dot(w_rh, h[t - 1]) + b_r
                prev = np.dot(w_rx, words[t])
                h[t] = np.tanh(current + prev)
            else:
                # t = 0 のとき、前の時刻の隠れ層を考慮しない
                current = np.dot(w_rx, words[t]) + b_r
                h[t] = np.tanh(current)
            # 各時刻の出力
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)
            # 各時刻の出力の最大値のインデックス (tuple(ndarray,) で返すので [0][0] を指定する必要がある)
            y[t] = np.where(p[t] == max(p[t]))[0][0]

        return h, p, y

    def gradient(self, words, h, p, p_correct):
        # 誤差情報の初期化
        self.init_error()

        # 勾配を計算するための各パラメータを取得
        dw_rx, dw_rh, db_r, dw_oh, db_o = self.err
        _, w_rh, w_oh, _, _ = self.net

        # 誤差を伝播させるために、時刻を遡る
        for t in range(len(words))[::-1]:
            delta_o_ = p_correct[t] - p[t]  # 出力層の誤り

            # 出力層の重み勾配
            dw_oh += np.outer(delta_o_, h[t])
            db_o += delta_o_

            # 各隠れ層の重み勾配を求める
            if t == len(words) - 1:
                # 前の隠れ層の誤差を考慮しない
                delta_r = np.dot(delta_o_, w_oh)
            else:
                # 前の隠れ層の誤差を伝播させる
                prev = np.dot(delta_r_, w_rh)
                current = np.dot(delta_o_, w_oh)
                delta_r = current + prev
            # tanh の勾配
            delta_r_ = delta_r * (1 - h[t] ** 2)

            # 隠れ層の重み勾配
            dw_rx += np.outer(delta_r_, words[t])
            db_r += delta_r_
            if t != 0:
                dw_rh += np.outer(delta_r_, h[t - 1])

        return (dw_rx, dw_rh, db_r, dw_oh, db_o)

    def update_weights(self, delta, lam=0.01):
        # 現在の重みとバイアス、更新するための各誤差を取得
        w_rx, w_rh, w_oh, b_r, b_o = self.net
        dw_rx, dw_rh, db_r, dw_oh, db_o = delta

        # 隠れ層の重みとバイアスの更新
        w_rx += lam * dw_rx
        w_rh += lam * dw_rh
        b_r += lam * db_r
        # 出力層の重みとバイアスの更新
        w_oh += lam * dw_oh
        b_o += lam * db_o

        self.net = [w_rx, w_rh, w_oh, b_r, b_o]

    def train(self):
        # input_file = "../files/test/05-train-input.txt"
        input_file = "../files/data/wiki-en-train.norm_pos"

        # ファイルから素性を作成する
        self.create_feature(input_file)
        # 重みとバイアスの初期化
        self.init_network()

        for _ in tqdm(range(self.epoch), desc="epoch"):
            for words, y_correct in tqdm(self.feat_lab, desc="feats"):
                h, p, _ = self.forward(words)
                delta = self.gradient(words, h, p, y_correct)
                self.update_weights(delta)

        return self.net, self.word_ids, self.pos_ids



class testRNN:
    def __init__(self, net, word_ids, pos_ids):
        self.net = net
        self.word_ids = word_ids
        self.pos_ids = pos_ids

        self.vocab_size = len(self.word_ids)
        self.pos_size = len(self.pos_ids)

    def create_onehot(self, id, size):
        v = np.zeros(size)
        v[id] = 1
        return v


    def softmax(self, vec):
        const = max(vec)
        exp_vec = np.exp(vec - const)
        sum_vec = np.sum(exp_vec)
        return exp_vec / sum_vec


    def forward(self, words):
        # words : 文の onehot ベクトル
        # 単語数
        vocab = len(words)
        # 時刻 t における各値
        h = [np.ndarray for _ in range(vocab)]  # 隠れ層の値 h[t]
        p = [np.ndarray for _ in range(vocab)]  # 出力の確率分布 p[t]
        y = [0 for _ in range(vocab)]

        w_rx, w_rh, w_oh, b_r, b_o = self.net
        for t in range(vocab):
            if t != 0:
                # 入力,前の時刻の両方の重みで現時刻の隠れ層の値を求める
                current = np.dot(w_rh, h[t - 1]) + b_r
                prev = np.dot(w_rx, words[t])
                h[t] = np.tanh(current + prev)
            else:
                # t = 0 のとき、前の時刻の隠れ層を考慮しない
                current = np.dot(w_rx, words[t]) + b_r
                h[t] = np.tanh(current)
            # 各時刻の出力
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)
            # 各時刻の出力の最大値のインデックス (tuple(ndarray,) で返すので [0][0] を指定する必要がある)
            y[t] = np.where(p[t] == max(p[t]))[0][0]

        return h, p, y

    def test(self):
        test_file = "../files/data/wiki-en-test.norm"

        for line in open(test_file, "r", encoding="utf-8"):
            words = line.strip().split()
            ws = [None for _ in range(len(words))]
            for i, word in enumerate(words):
                if word in self.word_ids:
                    ws[i] = self.create_onehot(self.word_ids[word], self.vocab_size)
                else:
                    ws[i] = np.zeros(self.vocab_size)
            h, p, y = self.forward(ws)
        
            for tag in y:
                for key, val in self.pos_ids.items():
                    if val == tag:
                        print(key, end=" ")
            print()

        


if __name__ == "__main__":
    epoch = 3
    node = 20

    rnn = trainRNN(epoch, node)
    net, word_ids, pos_ids = rnn.train()

    rnn = testRNN(net, word_ids, pos_ids)
    rnn.test()

"""
epoch = 3, node = 20
Accuracy: 79.95% (3648/4563)

Most common mistakes:
JJ --> NN       148
NNS --> NN      134
NNP --> NN      71
RB --> NN       55
VBG --> NN      38
VBN --> NN      33
CD --> NN       29
VBN --> JJ      28
VBP --> NN      21
IN --> WDT      16

"""