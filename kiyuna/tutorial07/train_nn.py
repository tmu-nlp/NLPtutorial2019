import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
np.random.seed(42)


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def dump(data, file_name):
    with open(f"./pickles/{file_name}.pkl", 'wb') as f_out:
        pickle.dump(data, f_out)


def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        phi[ids[f'UNI:{word}']] += 1
    return phi


def init_net(feat_num, layer_num, node_num):
    net = []
    w_0 = np.random.rand(node_num, feat_num) / 5 - 0.1
    b_0 = np.random.rand(1, node_num) / 5 - 0.1
    net.append((w_0, b_0))

    while len(net) < layer_num:
        w = np.random.rand(node_num, node_num) / 5 - 0.1
        b = np.random.rand(1, node_num) / 5 - 0.1
        net.append((w, b))

    w_o = np.random.rand(1, node_num) / 5 - 0.1
    b_o = np.random.rand(1, 1) / 5 - 0.1
    net.append((w_o, b_o))
    return net


def forward_nn(net, phi_0):
    """ #07 p26 """
    phi = [None] * (len(net) + 1)
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    return phi


def backward_nn(net, phi, label):
    """ #07 p31 """
    J = len(net)
    delta = np.zeros(J + 1, dtype=np.ndarray)
    delta[-1] = np.array([label - phi[J][0]])
    delta_p = np.zeros(J + 1, dtype=np.ndarray)
    for i in range(J, 0, -1):
        delta_p[i] = delta[i] * (1 - phi[i] ** 2).T
        w, _ = net[i - 1]
        delta[i - 1] = np.dot(delta_p[i], w)
    return delta_p


def update_weights(net, phi, delta_p, λ):
    """ #07 p33 """
    for i in range(len(net)):
        w, b = net[i]
        w += λ * np.outer(delta_p[i + 1], phi[i])
        b += λ * delta_p[i + 1]


def train_nn(train_path, layer_num=1, node_num=2, epoch_num=1, λ=0.1):
    """ #07 p34 """
    ids = defaultdict(lambda: len(ids))
    for line in open(train_path):
        _, sentence = line.strip().split('\t')
        for word in sentence.split():
            ids[f'UNI:{word}']

    ids = dict(ids)
    feat_label = []
    for line in open(train_path):
        label, sentence = line.strip().split('\t')
        label = int(label)
        phi = create_features(sentence, ids)
        feat_label.append((phi, label))

    net = init_net(len(ids), layer_num, node_num)
    for _ in tqdm(range(epoch_num)):
        for phi_0, label in tqdm(feat_label):
            phi = forward_nn(net, phi_0)
            delta_p = backward_nn(net, phi, label)
            update_weights(net, phi, delta_p, λ)
    message()

    dump(net, 'net')
    dump(ids, 'ids')
    return net, ids


if __name__ == '__main__':
    train_path = '../../data/titles-en-train.labeled'
    train_nn(train_path, layer_num=1, node_num=2, epoch_num=1, λ=0.1)
