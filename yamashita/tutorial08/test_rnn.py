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


def test_rnn():
    with open('rnn_net', 'rb') as net_file, open('x_ids', 'rb') as x_ids_file, open('y_ids', 'rb') as y_ids_file:
        net = pickle.load(net_file)
        x_ids = pickle.load(x_ids_file)
        y_ids = pickle.load(y_ids_file)

    ids_y = {v: k for k, v in y_ids.items()}
    with open('../../data/wiki-en-test.norm', 'r', encoding='utf-8') as test_file, open('rnn_ans', 'w', encoding='utf-8') as o_file:
        for line in test_file:
            x_list = []
            words = line.strip('\n').split()
            for word in words:
                if word.lower() in x_ids:
                    x_list.append(create_one_hot(
                        x_ids[word.lower()], len(x_ids)))
                else:
                    x_list.append(np.zeros((len(x_ids), 1)))

            h, p, y_list = forward_rnn(net, x_list)
            ans = []
            for y in y_list:
                ans.append(f'{words.pop(0)}_{ids_y[y]}')
            o_file.write(' '.join(ans))
            o_file.write('\n')


def main():
    test_rnn()


if __name__ == '__main__':
    main()
