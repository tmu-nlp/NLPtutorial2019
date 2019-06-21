from collections import defaultdict
from tqdm import tqdm
import numpy as np
import dill
import sys
import pickle


def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        phi[ids[f'UNI:{word}']] += 1
    return phi


def init_net(feat_num, layer_num, node_num):
    w_0 = np.random.rand(node_num, feat_num) / 5 - 0.1
    b_0 = np.random.rand(1, node_num) / 5 - 0.1

    net = [(w_0, b_0)]
    while len(net) < layer_num:
        w = np.random.rand(node_num, node_num) / 5 - 0.1
        b = np.random.rand(1, node_num) / 5 - 0.1
        net.append((w, b))

    w_o = np.random.rand(1, node_num) / 5 - 0.1
    b_o = np.random.rand(1, 1) / 5 - 0.1
    net.append((w_o, b_o))
    return net


def forward_nn(net, phi_0):
    phi = [0 for _ in range(len(net)+1)]
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i+1] = np.tanh(np.dot(w, phi[i])+b).T
    return phi


def backward_nn(net, phi, label):
    j = len(net)
    delta = np.zeros(j+1, dtype=np.ndarray)
    delta[-1] = np.array([label - phi[j][0]])
    delta_dash = np.zeros(j+1, dtype=np.ndarray)
    for i in range(j, 0, -1):
        delta_dash[i] = delta[i] * (1-np.square(phi[i])).T
        w, _ = net[i-1]
        delta[i-1] = np.dot(delta_dash[i], w)
    return delta_dash


def update_weights(net, phi, delta_dash, lambda_):
    for i in range(len(net)):
        w, b = net[i]
        w += lambda_ * np.outer(delta_dash[i+1], phi[i])
        b += lambda_ * delta_dash[i+1]


def train_nn(input_path, layer_num=1, node_num=2, epoch_num=1, lambda_=0.1):
    ids = defaultdict(lambda: len(ids))
    feat_label = []

    with open(input_path, 'r', encoding='utf-8') as i_file:
        for line in i_file:
            _, sentence = line.strip().split('\t')
            for word in sentence.split():
                ids[f'UNI:{word}']

    with open(input_path, 'r', encoding='utf-8') as i_file:
        for line in i_file:
            label, sentence = line.strip().split('\t')
            label = int(label)
            phi = create_features(sentence, ids)
            feat_label.append((phi, label))

    net = init_net(len(ids), layer_num, node_num)

    for _ in tqdm(range(epoch_num)):
        for phi_0, label in tqdm(feat_label):
            phi = forward_nn(net, phi_0)
            delta_dash = backward_nn(net, phi, label)
            update_weights(net, phi, delta_dash, lambda_)

    with open('net', 'wb') as net_file, open('ids', 'wb') as ids_file:
        print(len(net), len(ids))
        pickle.dump(net, net_file)
        pickle.dump(dict(ids), ids_file)


def main():
    train_input_path = '../../data/titles-en-train.labeled'
    train_nn(train_input_path)


if __name__ == '__main__':
    main()
