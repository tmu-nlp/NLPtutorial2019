import numpy as np
from collections import defaultdict
import dill
from tqdm import tqdm
xids = defaultdict(lambda: len(xids))  # word:index
yids = defaultdict(lambda: len(yids))  # pos:index
node = 20


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward_rnn(net, x_list):
    h = [0] * len(x_list)  # 隠れ層の値、次の時刻に影響する
    p = [0] * len(x_list)  # 出力の確率分布（すべて）
    y = [0 for _ in range(len(x_list))]  # 出力の確率分布、最も確率が高いもの
    w_rx = net[0]
    w_rh = net[1]
    w_oh = net[2]
    b_r = net[3]
    b_o = net[4]
    for t in range(len(x_list)):
        if t == 0:
            h[t] = np.tanh(np.dot(w_rx, x_list[t]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_rx, x_list[t]) + np.dot(w_rh, h[t-1]) + b_r)
        p[t] = softmax(np.dot(w_oh, h[t]) + b_o)
        y[t] = np.argmax(p[t])

    return h, p, y


def initialize_network():
    w_rx = np.random.rand(node, len(xids)) / 5 - 0.1  # 入力の重み
    w_rh = np.random.rand(node, node) / 5 - 0.1  # 次の時刻に伝播する重み
    w_oh = np.random.rand(len(yids), node) / 5 - 0.1  # 出力、softmaxに渡す重み
    b_r = np.random.rand(node) / 5 - 0.1  # 入力層のバイアス
    b_o = np.random.rand(len(yids)) / 5 - 0.1  # 出力層のバイアス

    return [w_rx, w_rh, w_oh, b_r, b_o]


def initialize_error():
    dw_rx = np.zeros((node, len(xids))) # 入力の重み
    dw_rh = np.zeros((node, node)) # 次の時刻に伝播する重み
    dw_oh = np.zeros((len(yids), node))  # 出力、softmaxに渡す重み
    db_r = np.zeros((node))  # 入力層のバイアス
    db_o = np.zeros((len(yids)))  # 出力層のバイアス

    return dw_rx, dw_rh, dw_oh, db_r, db_o


def gradient_rnn(net, x_list, h, p, y_correct):
    dw_rx, dw_rh, dw_oh, db_r, db_o = initialize_error()
    b_r = net[3]
    w_rh = net[1]
    w_oh = net[2]
    delta_r_ =np.zeros(len(b_r))
    for t in range(len(x_list))[::-1]:
        delta_o_ = y_correct[t] - p[t]
        dw_oh += np.outer(delta_o_, h[t])
        db_o += delta_o_

        delta_r = np.dot(delta_r_, w_rh) + np.dot(delta_o_, w_oh)
        delta_r_ = delta_r * (1 - h[t]**2)

        dw_rx += np.outer(delta_r_, x_list[t])
        db_r += delta_r_

        if t is not 0:
            dw_rh += np.outer(delta_r_, h[t-1])

    return dw_rx, dw_rh, db_r, dw_oh, db_o


def update_weights(net, dw_rx, dw_rh, db_r, dw_oh, db_o, lam):
    net[0] += lam * dw_rx
    net[1] += lam * dw_rh
    net[3] += lam * db_r
    net[2] += lam * dw_oh
    net[4] += lam * db_o


def one_hot_vec(index: int, size: int):
    vec = np.zeros(size)
    vec[index] = 1
    return vec


def create_feature(filename: str) -> list:
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            for word_pos in line:
                word = word_pos.split('_')[0]
                pos = word_pos.split('_')[1]
                xids[word]
                yids[pos]

    feat_lab = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            x_list = []
            y_list = []
            for n, word_pos in enumerate(line):
                word = word_pos.split('_')[0]
                pos = word_pos.split('_')[1]
                x_list.append(one_hot_vec(xids[word], len(xids)))
                y_list.append(one_hot_vec(yids[pos], len(yids)))
            feat_lab.append((x_list, y_list))

    return feat_lab


def main():
    # filename = '../../test//05-train-input.txt'
    filename = '../../data/wiki-en-train.norm_pos'
    feat_lab = create_featre(filename)
    net = initialize_network()

    # tqdm: for文の進行状況を表示
    epoch = 3
    for _ in tqdm(range(epoch), desc="epoch"):
        for x_list, y_correct in tqdm(feat_lab, desc="feat"):
            h, p, y_predict = forward_rnn(net, x_list)
            dw_rx, dw_rh, db_r, dw_oh, db_o = gradient_rnn(net, x_list, h, p, y_correct)
            update_weights(net, dw_rx, dw_rh, db_r, dw_oh, db_o, 0.01)

    with open('test_net', 'wb') as net_f, \
         open('test_xids', 'wb') as xids_f, \
         open('test_yids', 'wb') as yids_f:
        net_f.write(dill.dumps(net))
        xids_f.write(dill.dumps(xids))
        yids_f.write(dill.dumps(yids))


if __name__ == "__main__":
    main()
