from collections import defaultdict
from tqdm import tqdm
import numpy as np
import dill
import sys
import pickle
import random


def create_one_hot(id, size):
    vec = np.zeros((size, 1))
    vec[id] = 1
    return vec


def init_net(layer_num, x_ids, y_ids):
    w_rx = np.random.rand(layer_num, len(x_ids)) / 50 - 0.01
    w_rh = np.random.rand(layer_num, layer_num) / 50 - 0.01
    b_r = np.random.rand(layer_num, 1) / 50 - 0.01
    w_oh = np.random.rand(len(y_ids), layer_num) / 50 - 0.01
    b_o = np.random.rand(len(y_ids), 1) / 50 - 0.01

    # print(b_r)
    net = [w_rx, w_rh, b_r, w_oh, b_o]
    return net


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def find_best(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y


def forward_rnn(net, x):
    h = []
    p = []
    y = []
    w_rx, w_rh, b_r, w_oh, b_o = net
    for t in range(len(x)):
        if t > 0:
            h.append(np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r))
        else:
            h.append(np.tanh(np.dot(w_rx, x[t]) + b_r))

        p.append(softmax(np.dot(w_oh, h[t]) + b_o))
        y.append(find_best(p[t]))
    return h, p, y


def gradient_rnn(net, x, h, p, y_correct, layer_num, x_size, y_size):
    w_rx, w_rh, b_r, w_oh, b_o = net
    # print(b_r)
    dw_rx = np.zeros((layer_num, x_size))
    dw_rh = np.zeros((layer_num, layer_num))
    db_r = np.zeros((layer_num, 1))
    dw_oh = np.zeros((y_size, layer_num))
    db_o = np.zeros((y_size, 1))

    err_r_d = np.zeros((len(b_r), 1))

    for t in reversed(range(len(x))):
        err_o_d = y_correct[t] - p[t]
        dw_oh += np.outer(err_o_d, h[t])
        db_o += err_o_d
        err_r = np.dot(w_rh, err_r_d) + np.dot(w_oh.T, err_o_d)
        err_r_d = err_r * (1 - h[t]**2)
        dw_rx += np.outer(err_r_d, x[t])
        db_r += err_r_d
        if t != 0:
            dw_rh += np.outer(err_r_d, h[t-1])

    d_list = [dw_rx, dw_rh, db_r, dw_oh, db_o]
    return d_list


def update_weights(net, d_list, lambda_):
    for i in range(len(net)):
        net[i] += lambda_ * d_list[i]
    return net


def train_rnn(input_path, layer_num=5, epoch_num=10, lambda_=0.01):
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    feat_label = []

    with open(input_path, 'r', encoding='utf-8') as t_file:
        for line in t_file:
            pairs = line.strip('\n').split()
            for pair in pairs:
                word, label = pair.split('_')
                x_ids[word]
                y_ids[label]

    x_size = len(x_ids)
    y_size = len(y_ids)

    with open(input_path, 'r', encoding='utf-8') as i_file:
        for line in i_file:
            x_list = []
            y_list = []
            pairs = line.strip('\n').split()
            for pair in pairs:
                word, label = pair.split('_')
                x_list.append(create_one_hot(x_ids[word], x_size))
                y_list.append(create_one_hot(y_ids[label], y_size))
            feat_label.append((x_list, y_list))

    net = init_net(layer_num, x_ids, y_ids)

    for _ in tqdm(range(epoch_num)):
        random.shuffle(feat_label)
        for x, y_correct in tqdm(feat_label):
            h, p, y_predict = forward_rnn(net, x)
            delta_list = gradient_rnn(
                net, x, h, p, y_correct, layer_num, x_size, y_size)
            net = update_weights(net, delta_list, lambda_)

    with open('rnn_net', 'wb') as net_file, open('x_ids', 'wb') as x_ids_file, open('y_ids', 'wb') as y_ids_file:
        print(len(net), x_size, y_size)
        pickle.dump(net, net_file)
        pickle.dump(dict(x_ids), x_ids_file)
        pickle.dump(dict(y_ids), y_ids_file)


def main():
    train_input_path = '../../data/wiki-en-train.norm_pos'
    train_rnn(train_input_path)


if __name__ == '__main__':
    main()
