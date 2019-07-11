import dill
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def one_hot_vec(index: int, size: int):
    vec = np.zeros((size))
    vec[index] = 1
    return vec


def forward_rnn(net, x_list):
    h = [0] * len(x_list)  # 隠れ層の値、次の時刻に影響する
    p = [0] * len(x_list) # 力の確率分布（すべて）
    y = [0] * len(x_list) # 出力の確率分布、最も確率が高いもの
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


def main():
    with open('test_net', 'rb') as net_f, \
         open('test_xids', 'rb') as xids_f,\
         open('test_yids', 'rb') as yids_f:
        net = dill.loads(net_f.read())
        xids = dill.loads(xids_f.read())
        yids = dill.loads(yids_f.read())

    filename = '../../data/wiki-en-test.norm'
    # filename = '../../test/05-test-input.txt'
    with open(filename, 'r') as f,\
         open('answer.txt', 'w') as out_f:
        for line in f:
            line = line.strip().split(' ')
            
            x_list = []
            for i, word in enumerate(line):
                if word in xids:
                    x = one_hot_vec(xids[word], len(xids))
                else:
                    x = np.zeros(len(xids))
                x_list.append(x)

            h, p, y_list = forward_rnn(net, x_list)
            answer = []
            for y in y_list:
                for pos, ids in yids.items():
                    if ids == y:
                        answer.append(pos)

            print(' '.join(answer), file=out_f)


if __name__ == "__main__":
    main()


# Accuracy: 79.42% (3624/4563)
#
# Most common mistakes:
# JJ --> NN       155
# NNS --> NN      148
# NNP --> NN      71
# RB --> NN       56
# VBN --> NN      46
# VBG --> NN      41
# VBN --> JJ      28
# CD --> NN       28
# VBP --> NN      23
# VBZ --> NN      14
