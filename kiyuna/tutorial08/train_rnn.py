import os
import sys
import numpy as np
import pickle
import random
from collections import defaultdict
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
np.random.seed(42)
random.seed(42)


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def dump(data, file_name):
    with open(f"./pickles/{file_name}.pkl", 'wb') as f_out:
        pickle.dump(data, f_out)


def create_one_hot(id, size):
    """ #08 p12 """
    vec = np.zeros((size, 1))
    vec[id] = 1
    return vec


def init_net(node_num, x_ids, y_ids):
    w_rx = np.random.rand(node_num, len(x_ids)) / 50 - 0.01
    w_rh = np.random.rand(node_num, node_num) / 50 - 0.01
    b_r = np.random.rand(node_num, 1) / 50 - 0.01
    w_oh = np.random.rand(len(y_ids), node_num) / 50 - 0.01
    b_o = np.random.rand(len(y_ids), 1) / 50 - 0.01

    net = [w_rx, w_rh, b_r, w_oh, b_o]
    return net


def softmax(x):
    """ #08 p9 """
    r = np.exp(x)
    return r / r.sum()


def find_max(p):
    """ #08 p10 """
    return np.argmax(p)
    y = 0
    for i in range(1, len(p)):
        if p[i] > p[y]:
            y = i
    return y


def forward_rnn(net, x):
    """ #08 p16 """
    w_rx, w_rh, b_r, w_oh, b_o = net
    h = [None] * len(x)
    p = [None] * len(x)
    y = [None] * len(x)
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t - 1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)
        p[t] = softmax(np.dot(w_oh, h[t]) + b_o)
        y[t] = find_max(p[t])
    return h, p, y


def gradient_rnn(net, x, h, p, y_correct, x_size, y_size):
    """ #08 p30 """
    w_rx, w_rh, b_r, w_oh, b_o = net
    dw_rx, dw_rh, db_r, dw_oh, db_o = [np.zeros_like(x) for x in net]

    err_d_r_p = np.zeros((len(b_r), 1))

    for t in range(len(x) - 1, -1, -1):
        err_d_o = y_correct[t] - p[t]
        dw_oh += np.outer(err_d_o, h[t])
        db_o += err_d_o
        err_d_r = np.dot(w_rh, err_d_r_p) + np.dot(w_oh.T, err_d_o)
        err_d_r_p = err_d_r * (1 - h[t]**2)
        dw_rx += np.outer(err_d_r_p, x[t])
        db_r += err_d_r_p
        if t != 0:
            dw_rh += np.outer(err_d_r_p, h[t - 1])

    d_list = [dw_rx, dw_rh, db_r, dw_oh, db_o]
    return d_list


def update_weights(net, d_list, λ):
    """ #08 p31 """
    for w, dw in zip(net, d_list):
        w += λ * dw
    return net


def train_rnn(train_path, node_num=5, epoch_num=10, λ=0.01):
    """ #08 p32 """
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    for line in open(train_path):
        pairs = line.split()
        for pair in pairs:
            word, label = pair.split('_')
            x_ids[word]
            y_ids[label]

    x_ids, x_size = dict(x_ids), len(x_ids)
    y_ids, y_size = dict(y_ids), len(y_ids)
    feat_label = []
    for line in open(train_path):
        x_list = []
        y_list = []
        pairs = line.split()
        for pair in pairs:
            word, label = pair.split('_')
            x_list.append(create_one_hot(x_ids[word], x_size))
            y_list.append(create_one_hot(y_ids[label], y_size))
        feat_label.append((x_list, y_list))

    net = init_net(node_num, x_ids, y_ids)

    for _ in tqdm(range(epoch_num)):
        random.shuffle(feat_label)
        for x, y_correct in tqdm(feat_label):
            h, p, y_predict = forward_rnn(net, x)
            delta_list = gradient_rnn(
                net, x, h, p, y_correct, x_size, y_size)
            net = update_weights(net, delta_list, λ)
    message()

    dump(net, 'rnn_net')
    dump(x_ids, 'x_ids')
    dump(y_ids, 'y_ids')
    return net, x_ids, y_ids


if __name__ == '__main__':
    train_path = '../../data/wiki-en-train.norm_pos'
    train_rnn(train_path, node_num=5, epoch_num=10, λ=0.01)
