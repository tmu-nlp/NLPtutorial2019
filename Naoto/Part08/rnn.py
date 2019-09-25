import numpy as np
import subprocess
import sys
from collections import defaultdict
from tqdm import tqdm
import os
import pickle

np.random.seed(42)


class RNN:
    def __init__(self, H, epoch=1):
        self.H = H
        self.epoch = epoch
        self.x_ids = defaultdict(lambda: len(self.x_ids))
        self.y_ids = defaultdict(lambda: len(self.y_ids))
        self.x_len = 0
        self.y_len = 0

    def init_net(self):
        # w_rx = np.random.rand(self.x_len, self.H) / 5 - 0.1
        w_rx = np.random.rand(self.H, self.x_len) / 5 - 0.1
        w_rh = np.random.rand(self.H, self.H) / 5 - 0.1
        b_r = np.random.rand(self.H, 1) / 5 - 0.1
        # w_oh = np.random.rand(self.H, self.y_len) / 5 - 0.1
        w_oh = np.random.rand(self.y_len, self.H) / 5 - 0.1
        b_o = np.random.rand(self.y_len, 1) / 5 - 0.1
        self.net = [w_rx, w_rh, b_r, w_oh, b_o]

    def softmax(self, x):
        """ #08 p9 """
        r = np.exp(x)
        return r / r.sum()

    def find_best(self, p):
        """ #8 10 """
        return np.argmax(p)
        y = 0
        for i in range(1, len(p)):
            if p[i] > p[y]:
                y = i
        return y

    def create_one_hot(self, id_, size):
        """ #8 p12 """
        vec = np.zeros((size, 1))
        if id_ is not None:
            vec[id_] = 1
        return vec

    def forward(self, x):
        """ #8 p16 """
        w_rx, w_rh, b_r, w_oh, b_o = self.net
        x_len = len(x)
        h = [None for _ in range(x_len)]  # 隠れ層の値 (各時間tにおいて)
        p = [None for _ in range(x_len)]  # 出力の確率分布の値 (各時間tにおいて)
        y = [None for _ in range(x_len)]  # 出力の確率分布の値 (各時間tにおいて)
        for t in range(x_len):
            if t > 0:
                h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t - 1]) + b_r)   # w_rx → w_rx.T
            else:
                h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)    # w_rx → w_rx.T
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)        # w_oh → w_oh.T
            y[t] = self.find_best(p[t])
        return h, p, y

    def gradient(self, x, h, p, y_):
        """ #8 p30 """
        w_rx, w_rh, b_r, w_oh, b_o = self.net
        Δw_rx, Δw_rh, Δb_r, Δw_oh, Δb_o = [np.zeros_like(x) for x in self.net]

        δ_r_ = np.zeros((len(b_r), 1))   # 次の時間から伝搬するエラー
        for t in range(len(x) - 1, -1, -1):
            δ_o_ = y_[t] - p[t]                                # 出力層エラー              # 出力層重み勾配
            Δw_oh += np.outer(δ_o_, h[t])                  # 出力層重み勾配
            Δb_o += δ_o_                                   # 出力層重み勾配
            # δ_r = np.dot(δ_r_, w_rh) + np.dot(w_oh, δ_o_)   # 逆伝搬    np.dot(δ_o_, w_oh) → np.dot(w_oh, δ_o_)
            δ_r = np.dot(w_rh, δ_r_) + np.dot(w_oh.T, δ_o_)   # 逆伝搬    np.dot(δ_o_, w_oh) → np.dot(w_oh, δ_o_)
            δ_r_ = δ_r * (1 - h[t] ** 2)                    # tanh の勾配
            # Δw_rx += np.outer(x[t], δ_r_)                   # 隠れ層重み勾配
            Δw_rx += np.outer(δ_r_, x[t])                   # 隠れ層重み勾配
            Δb_r += δ_r_                                    # 隠れ層重み勾配
            if t != 0:
                Δw_rh += np.outer(δ_r_, h[t - 1])
                # Δw_rh += np.outer(h[t - 1], δ_r_)
        Δ = [Δw_rx, Δw_rh, Δb_r, Δw_oh, Δb_o]
        return Δ

    def update_weights(self, Δ, λ=0.01):
        w_rx, w_rh, b_r, w_oh, b_o = self.net
        Δw_rx, Δw_rh, Δb_r, Δw_oh, Δb_o = Δ
        w_rx += λ * Δw_rx
        w_rh += λ * Δw_rh
        b_r += λ * Δb_r
        w_oh += λ * Δw_oh
        b_o += λ * Δb_o
        self.net = w_rx, w_rh, b_r, w_oh, b_o

    def dump(self, data, file_name):
        os.makedirs('pickles_naoto', exist_ok=True)
        with open(f"pickles_naoto/{file_name}.pkl", 'wb') as f_out:
            pickle.dump(data, f_out)

    def train(self, train_path):
        """ #8 p32 """
        X, Y_corrrect = [], []
        for line in map(lambda x: x.rstrip(), open(train_path)):
            words, tags = map(list, zip(*map(lambda x: x.split('_'), line.split())))
            X.append([])
            Y_corrrect.append([])
            for word, tag in zip(words, tags):
                X[-1].append(self.x_ids[word])
                Y_corrrect[-1].append(self.y_ids[tag])
        self.x_ids = dict(self.x_ids)
        self.y_ids = dict(self.y_ids)
        self.x_len = len(self.x_ids)
        self.y_len = len(self.y_ids)
        self.init_net()

        x_onehot_list = []
        y_onehot_list = []
        for x, y_correct in zip(X, Y_corrrect):
            x_onehot_list.append([])
            y_onehot_list.append([])
            for i in range(len(x)):
                x_onehot_list[-1].append(self.create_one_hot(x[i], self.x_len))
            for i in range(len(y_correct)):
                y_onehot_list[-1].append(
                    self.create_one_hot(y_correct[i], self.y_len))
        for _ in tqdm(range(self.epoch)):
            for x_onehot, y_onehot in zip(x_onehot_list, y_onehot_list):
                h, p, y_predict = self.forward(x_onehot)
                Δ = self.gradient(x_onehot, h, p, y_onehot)
                self.update_weights(Δ)
        self.dump(self.net, 'rnn_net')
        self.dump(self.x_ids, 'x_ids')
        self.dump(self.y_ids, 'y_ids')

    def test(self, test_path, out_path):
        self.y_ids_recover = {value: key for key, value in self.y_ids.items()}
        with open(out_path, 'w') as f:
            x_onehot_list = []
            for line in map(lambda x: x.rstrip(), open(test_path)):
                words = line.split(' ')
                x_onehot_list.append([])
                for word in words:
                    if word in self.x_ids:
                        x_onehot_list[-1].append(self.create_one_hot(
                            self.x_ids[word], self.x_len))
                    else:
                        x_onehot_list[-1].append(self.create_one_hot(
                            None, self.x_len))
            for x_onehot in x_onehot_list:
                h, p, y = self.forward(x_onehot)
                predict_list = []
                for predict_num in y:
                    predict = self.y_ids_recover[predict_num]
                    predict_list.append(predict)
                print(' '.join(predict_list), file=f)


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        train_path = '../../../nlptutorial/test/05-train-input.txt'
        test_path = '../../../nlptutorial/test/05-test-input.txt'
        ans_path = '../../../nlptutorial/test/05-test-answer.txt'
    else:
        train_path = '../../../nlptutorial/data/wiki-en-train.norm_pos'
        test_path = '../../../nlptutorial/data/wiki-en-test.norm'
        ans_path = '../../../nlptutorial/data/wiki-en-test.pos'
    out_path = 'out.txt'
    script_path = '../../../nlptutorial/script/gradepos.pl'

    rnn = RNN(5, 20)
    rnn.train(train_path)
    rnn.test(test_path, out_path)
    subprocess.run(f'perl {script_path} {ans_path} {out_path}'.split())
