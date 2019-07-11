from collections import defaultdict
import numpy as np
import dill
from tqdm import tqdm
import random

epoch = 50
hidden_node = 100
lam = 0.01

def create_one_hot(id, size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec

def find_max(p):
    y = 0
    for i in range(1, len(p)):
        if p[i] > p[y]:
            y = i
    return y

def initialize_net_randomly(vocab_size, pos_size):
    w_rx = (np.random.rand(hidden_node, vocab_size) - 0.5)/5
    b_r = (np.random.rand(hidden_node) - 0.5)/5
    w_rh = (np.random.rand(hidden_node, hidden_node) - 0.5)/5
    w_oh = (np.random.rand(pos_size, hidden_node) - 0.5)/5
    b_o = (np.random.rand(pos_size) - 0.5)/5
    net = (w_rx, b_r, w_rh, w_oh, b_o)
    return net

def initialize_delta(vocab_size, pos_size):
    dw_rx = np.zeros((hidden_node, vocab_size))
    dw_rh = np.zeros((hidden_node, hidden_node))
    db_r = np.zeros(hidden_node)
    dw_oh = np.zeros((pos_size, hidden_node))
    db_o = np.zeros(pos_size)
    delta = (dw_rx, dw_rh, db_r, dw_oh, db_o)
    return delta

def forward_rnn(net, x):
    w_rx, b_r, w_rh, w_oh, b_o = net
    h = [np.ndarray for _ in range(len(x))]
    p = [np.ndarray for _ in range(len(x))]
    y = [np.ndarray for _ in range(len(x))]
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t - 1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)
        p[t] = np.tanh(np.dot(w_oh, h[t]) + b_o)
        y[t] = find_max(p[t])
    return (h, p, y)

def gradient_rnn(net, x, h, p, y_prime, vocab_size, pos_size):
    delta = initialize_delta(vocab_size, pos_size)
    dw_rx, dw_rh, db_r, dw_oh, db_o, = delta
    w_ri, b_r, w_rh, w_oh, b_o = net
    delta_r_prime = np.zeros(len(b_r))
    for t in range(len(x) - 1, -1, -1):
        delta_o_prime = y_prime[t] - p[t]
        dw_oh += np.outer(delta_o_prime, h[t])
        db_o += delta_o_prime
        if t == len(x) - 1:
            delta_r = np.dot(delta_o_prime, w_oh)
        else:
            delta_r = np.dot(delta_r_prime, w_rh) + np.dot(delta_o_prime, w_oh)
        delta_r_prime = delta_r*(1 - h[t]**2)
        dw_rx += np.outer(delta_r_prime, x[t])
        db_r += delta_r_prime
        if t != 0:
            dw_rh += np.outer(delta_r_prime, h[t - 1])
    return (dw_rx, dw_rh, db_r, dw_oh, db_o)

def update_weights(net, delta, lam):
    dw_rx, dw_rh, db_r, dw_oh, db_o = delta
    w_rx, b_r, w_rh, w_oh, b_o = net
    w_rx += lam*dw_rx
    w_rh += lam*dw_rh
    b_r += lam*db_r
    w_oh += lam*dw_oh
    b_o += lam*db_o

if __name__ == "__main__":
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    feat_label = []
    # "../../data/wiki-en-test.norm_pos"
    with open("../../test/05-train-input.txt", 'r') as train_file:
        for line in train_file:
            word_tags = line.rstrip().split(' ')
            for word_tag in word_tags:
                word, tag = word_tag.split('_')
                x_ids[word]
                y_ids[tag]
    with open("../../test/05-train-input.txt", 'r') as train_file:
        for line in train_file:
            words = []
            tags = []
            word_tags = line.rstrip().split(' ')
            for word_tag in word_tags:
                word, tag = word_tag.split('_')
                words.append((create_one_hot(x_ids[word], len(x_ids))))
                tags.append((create_one_hot(y_ids[tag], len(y_ids))))
            feat_label.append((words, tags))
    net = initialize_net_randomly(len(x_ids), len(y_ids))

    for _ in tqdm(range(epoch)):
        for x, y_prime in feat_label:
            h, p, y = forward_rnn(net, x)
            delta = gradient_rnn(net, x, h, p, y_prime, len(x_ids), len(y_ids))
            update_weights(net, delta, lam)

    with open("net", "wb") as net_file, open("x_ids", "wb") as xids_file, open("y_ids", "wb") as yids_file:
        dill.dump(net, net_file)
        dill.dump(x_ids, xids_file)
        dill.dump(y_ids, yids_file)
