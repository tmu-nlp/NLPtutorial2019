from collections import defaultdict
import numpy as np
import dill

train_path = '../../data/titles-en-train.labeled'

epoch = 10
lam = 0.03
layer = 3
hidden_node = 4

def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split(' ')
    for word in words:
        phi[ids[f"UNI:{word}"]] += 1
    return phi

def initialize_net_randomly(ids):
    net = []
    # 入力層 -0.1以上0.1未満の値で初期化
    w_in = (np.random.rand(hidden_node, len(ids)) - 0.5)/5 # サイズ (hidden_node)*(len(ids))
    b_in = (np.random.rand(hidden_node) - 0.5)/5 # サイズ (hidden_node)*1
    net.append((w_in, b_in))
    # 隠れ層
    for _ in range(layer - 2):
        w = (np.random.rand(hidden_node, hidden_node) - 0.5)/5 # サイズ (hidden_node)*(hidden_node)
        b = (np.random.rand(hidden_node) - 0.5)/5 # サイズ (hidden_node)*1
        net.append((w, b))
    # 出力層
    w_out = (np.random.rand(1, hidden_node) - 0.5)/5 # サイズ 1*(hidden_node)
    b_out = (np.random.rand(1) - 0.5)/5 # サイズ 1*1
    return net

def forward_nn(net, phi_zero):
    # phisは各層の出力の値をもつ
    phis = [phi_zero]
    for i in range(len(net)):
        w, b = net[i]
        phis.append(np.tanh(np.dot(w, phis[i]) + b))
    return phis

def backward_nn(net, phis, label):
    J = len(net)
    delta = [np.ndarray for _ in range(J)] # サイズ J*1
    delta.append(label - phis[J - 1]) # サイズ (J + 1)*1
    delta_ = [np.ndarray for _ in range(J + 1)] # サイズ (J + 1)*1
    for i in range(J - 1, -1, -1):
        delta_[i + 1] = delta[i + 1]*(1 - phis[i + 1]**2)
        w, b = net[i]
        delta[i] = np.dot(delta_[i + 1], w)
    return delta_

def update_weights(net, phis, delta_, lam):
    for i in range(len(net)):
        w, b = net[i]
        w += lam*np.outer(delta_[i + 1], phis[i])
        b += lam*delta_[i + 1]

if __name__ == '__main__':
    ids = defaultdict(lambda: len(ids))
    feat_label = []

    with open(train_path, 'r') as f:
        train = []
        for line in f:
            line = line.rstrip()
            label, sentence = line.split('\t')
            train.append((int(label), sentence))
            for word in sentence.split(' '):
                ids[f"UNI:{word}"]

    for label, sentence in train:
        feat_label.append((create_features(sentence, ids), label))

    net = initialize_net_randomly(ids)

    for _ in range(epoch):
        error = 0
        for phi_zero, label in feat_label:
            phis = forward_nn(net, phi_zero)
            delta_ = backward_nn(net, phis, label)
            update_weights(net, phis, delta_, lam)
            error += abs(label - phis[len(net)][0])

    with open('network', 'wb') as f_network, open('ids', 'wb') as f_ids:
        dill.dump(net, f_network)
        dill.dump(ids, f_ids)











'''
from collections import defaultdict
import numpy as np

ids = defaultdict(lambda: len(ids))
feat_lab = []

def create_features(x):
    phi_tmp = defaultdict(lambda: 0)
    words = x.split(' ')
    for word in words:
        phi_tmp[ids["UNI:" + word]] += 1
    phi = [0]*len(ids)
    for key, value in phi_tmp.items():
        phi[key] = value
    phi = np.array(phi)
    return phi

def forward_nn(network, phi_0):
    phi = []*len(network)
    phi[0] = phi_0
    for i in range(1, len(neteork)):
        w, b = network[i - 1]
        phi[i] = np.tanh(np.dot(w, phi[i - 1]) + b)
    return phi

def backward_nn(net, phi, y_prime):
    J = len(net)
    delta = np.zeros(J + 1)
    delta[-1] = np.array([y_prime - phi[J][0]])
    delta_prime = np.zeros(J + 1)
    for i in range(J - 1, -1, -1):
        delta_prime[i + 1] = delta[i + 1] * (1 - np.square(phi[i]))
        w, b = net[i]
        delta[i] = np.dot(delta_prime[i + 1], w)
    return delta_prime

def update_weights(net, phi, delta_prime, lam):
    for i in range(0, len(net)):
        w, b = net[i]
        w += lam*np.outer(delta_prime[i + 1], phi[i])
        b += lam*delta_prime[i + 1]

input_file_path = "../../test/03-train-input.txt"

with open(input_file_path, 'r') as input_file:
    for line in input_file:
        line = line.strip().split('\t')
        x = line[1]
        y = int(line[0])
        feat_lab.append([create_features(x), y])

net = []
'''
