import numpy as np
from collections import defaultdict
from tqdm import tqdm


array = np.ndarray

class TrainRNN:
    def __init__(self, node: int, epoch: int):
        # hyper prameter
        self.node = node
        self.epoch = epoch

        self.word_ids = defaultdict(lambda: len(self.word_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        self.net = []
        self.feat_lab = []

        self.vocab_size = 0
        self.pos_size = 0


    def initialize_network(self):
        # np.random.rand(n, m) : 0 ~ 1 未満の n 行 m 列の一様乱数のリストを生成
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
    
    def create_ids(self, path: str):
        for line in open(path, 'r',encoding='utf-8'):
            for word_pos in line.strip().split():
                word, pos = word_pos.split('_')
                self.word_ids[word]
                self.pos_ids[pos]
        
        self.vocab_size = len(self.word_ids)
        self.pos_size = len(self.pos_ids)

        for line in open(path, 'r', encoding='utf-8'):
            word_pos_list = line.strip().split()
            #文ごとにonehotを作る
            ws = [None for _ in range(len(word_pos_list))]
            ps = [None for _ in range(len(word_pos_list))]

            for i, word_pos in enumerate(word_pos_list):
                word, pos = word_pos.split('_')
                ws[i] = self.create_onehot(self.word_ids[word], self.vocab_size)
                ps[i] = self.create_onehot(self.pos_ids[pos], self.pos_size)
            self.feat_lab.append((ws, ps))


    def create_onehot(self, id, size):
        v = np.zeros(size)
        v[id] = 1
        return v


    def forward(self, onehot):
        # 単語数
        vocab = len(onehot)

        # 隠れ層の値
        h = [np.ndarray for _ in range(vocab)]
        # 出力の確率分布
        p = [np.ndarray for _ in range(vocab)]
        y = [0 for _ in range(vocab)]

        w_rx, w_rh, w_oh, b_r, b_o = self.net

        for t in range(vocab):
            if t != 0:
                # 入力、前の時刻の両方の重みで現時刻の隠れ層の値を求める
                current = np.dot(w_rh, h[t-1]) + b_r
                prev = np.dot(w_rx, onehot[t])
                h[t] = np.tanh(current + prev)
            else:
                current = np.dot(w_rx, onehot[t]) + b_r
                h[t] = np.tanh(current)
            
            # 各時刻の出力
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)
            # 各時刻の出力の最大値のインデックス (tuple(ndarray,) で返すので [0][0] を指定する必要がある)
            y[t] = np.where(p[t] == max(p[t]))[0][0]

        return h, p, y    

    def softmax(self, vec):
        const = max(vec)
        exp_vec = np.exp(vec - const)
        sum_vec = np.sum(exp_vec)
        return exp_vec / sum_vec


    def update_weights(self, delta, lam=0.1):
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


    def prepare_training(self, train_file: str):
        data = []
        for line in open(train_file, "r", encoding="utf-8"):
            label, sentence = line.rstrip().split("\t")
            data.append((int(label), sentence))
            for word in sentence.split():
                self.ids[f"UNI:{word}"]
        self.feat_lab = [(self.create_features(sent), lab) for lab, sent in data]



    def train(self, path):
        # ファイルを読み込み、素性を作る
        self.create_ids(path)

        # ネットワークを一様乱数で初期化
        self.initialize_network()

        # train
        for _ in tqdm(range(self.epoch), desc='epoch'):
            for words, p_correct in tqdm(self.feat_lab, desc='feats'):
                h, p, _ = self.forward(words)
                delta = self.gradient(words, h, p, p_correct)
                self.update_weights(delta)
        return self.net, self.word_ids, self.pos_ids


    def gradient(self, words, h, p, p_correct):
        # 誤差の初期化
        self.init_error()

        # 勾配を計算するためのパラメータ
        dw_rx, dw_rh, db_r, dw_oh, db_o = self.err
        _, w_rh, w_oh, _, _ = self.net

        # 誤差の伝搬(さかのぼる)
        for t in range(len(words))[::-1]:
            delta_o_ = p_correct[t] - p[t]  # 出力層の誤り

            # 出力層の重み勾配
            dw_oh += np.outer(delta_o_, h[t])
            db_o += delta_o_

            # 各隠れ層の重み勾配を求める
            if t == len(words) - 1:
                delta_r = np.dot(delta_o_, w_oh)
            else:
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
            
    def init_error(self):
        # 入力層, 隠れ層の重み勾配
        dw_rx = np.zeros((self.node, self.vocab_size))
        dw_rh = np.zeros((self.node, self.node))
        db_r = np.zeros(self.node)

        # 出力層の重み勾配
        dw_oh = np.zeros((self.pos_size, self.node))
        db_o = np.zeros(self.pos_size)

        self.err = [dw_rx, dw_rh, db_r, dw_oh, db_o]



if __name__ == "__main__":

    # hyper parameters
    epoch = 3
    node = 20

    # train
    rnn = TrainRNN(node, epoch)
    net, word_ids, pos_ids = rnn.train()


"""
layer = 2, node = 2, epoch = 1
Accuracy = 91.675522%
layer = 1, node = 2, epoch = 1
Accuracy = 91.108750%
"""