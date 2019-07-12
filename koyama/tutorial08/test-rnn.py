import numpy as np
import dill

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

def forward_rnn(net, x):
    w_rx, b_r, w_rh, w_oh, b_o = net
    h = [np.ndarray for _ in range(len(x))]
    p = [np.ndarray for _ in range(len(x))]
    y = [np.ndarray for _ in range(len(x))]
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r);
        p[t] = np.tanh(np.dot(w_oh, h[t]) + b_o)
        y[t] = find_max(p[t])
    return (h, p, y)

if __name__ == '__main__':
    with open("net", "rb") as net_file, open("x_ids", "rb") as xids_file, open("y_ids", "rb") as yids_file:
        net = dill.load(net_file)
        x_ids = dill.load(xids_file)
        y_ids = dill.load(yids_file)
    # '../../data/wiki-en-test.norm'
    with open("../../test/05-test-input.txt", 'r') as test_file, open("my_answer.pos", "w") as answer_file:
        for line in test_file:
            x_list = []
            words = line.rstrip().split(' ')
            for word in words:
                if word in x_ids:
                    x_list.append(create_one_hot(x_ids[word], len(x_ids)))
                else:
                    x_list.append(np.zeros(len(x_ids)))
            h, p, y_list = forward_rnn(net, x_list)
            for tag in y_list:
                for key, value in y_ids.items():
                    if value == tag:
                        print(key, end=' ', file=answer_file)
            print(file=answer_file)

'''
epoch = 50, hidden_node = 100, lambda = 0.01 のとき
Accuracy: 98.29% (4485/4563)

Most common mistakes:
JJS --> NN      6
JJR --> NN      4
PRP --> CD      4
JJR --> RBR     4
PDT --> DT      3
VBG --> NN      3
RB --> IN       3
JJS --> RBS     3
VBP --> VB      3
PDT --> JJ      3

HMMのとき
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBP --> VB      7
'''
