import numpy as np
from collections import defaultdict
from collections import Counter
import dill
ids = defaultdict(lambda: len(ids))  # word:index


def create_feature(x: str) -> dict:
    phi = defaultdict(int)
    words = x.split(' ')
    for word in words:
        phi[ids['UNI:'+word]] += 1

    return phi


def initialize_network(node: int):
    net = []

    w1 = np.random.rand(node, len(ids)) / 5 - 0.1
    b1 = np.random.rand(node, 1) / 5 - 0.1
    net.append((w1, b1))

    w2 = np.random.rand(1, node) / 5 - 0.1
    b2 = np.random.rand(1, 1) / 5 - 0.1
    net.append((w2, b2))

    return net


def forward_nn(net, phi0):
    phi = [phi0]

    for i, node in enumerate(net):
        w, b = node[0], node[1]
        phi.append(np.tanh(np.dot(w, phi[i]) + b))

    return phi


def backward_nn(net, phi, y):
    j = len(net)
    delta = [y - phi[j][0]]
    delta_ = []
    for i in range(j):
        # print(f'delta{j-i} = {delta[i]}')
        # print(f'phi{j-i} = {phi[j-i]}')
        delta_.append(delta[i] * (1 - phi[j-i]**2))
        w, b = net[j-i-1]
        # print(f'w{j-i} = {w}')
        # print(f'delta_{j-i} = {delta_[i]}')
        delta.append(np.dot(w.T, delta_[i]))

    return delta_


def update_weights(net, phi, delta_, lam):
    for i in range(len(net)):
        w, b = net[i]
        w += lam * np.outer(delta_[i], phi[i])
        b += lam * delta_[i]
        net[i] = w, b


def main():
    feat_lab = []
    with open('../../data/titles-en-train.labeled', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            y = int(line[0])
            x = line[1]
            phi0 = []
            for count in sorted(create_feature(x).items()):
                phi0.append([count[1]])
            feat_lab.append((create_feature(x), y))
        net = initialize_network(2)


    for n, feat in enumerate(feat_lab):
        phi_dict = feat[0]
        y = feat[1]
        phi0 = [[0]] * len(ids)
        for index, value in phi_dict.items():
            phi0[index] = [value]
        phi = forward_nn(net, phi0)
        # print(phi)
        # print(phi)
        delta_ = backward_nn(net, phi, y)
        delta_.reverse()
        update_weights(net, phi, delta_, 0.1)

        if n % 1000 == 0:
            print(f'{n}行目')

    with open('net', 'wb') as net_f, \
         open('ids', 'wb') as ids_f:
        net_f.write(dill.dumps(net))
        ids_f.write(dill.dumps(ids))
        

if __name__ == "__main__":
    main()
