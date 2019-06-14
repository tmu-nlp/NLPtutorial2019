import sys
import math
from collections import defaultdict
import numpy as np


def update_weight(w: dict, phi: dict, y: int):
    c = 0.001
    for word, weight in w.items():
        if abs(weight) < c:
            w[word] = 0
        else:
            w[word] -= np.sign(weight) * c

    for word, count in phi:
        w[word] += count * y


def main():
    w = defaultdict(lambda:0)
    margin = 0
    c = 0.001
    with open('../../data/titles-en-train.labeled', 'r') as f:
        for i, line in enumerate(f):
            # print(line)
            phi = defaultdict(lambda:0)
            line = line.strip().split('\t')
            y = int(line[0])
            x = line[1]
            words = x.split(' ')
            # print(words)

            # phi(単語の出現数)の計算
            for word in words:
                phi[word] += 1
            # print(phi)

            val = 0
            for word, count in phi.items():
                val += w[word] * count
            val = val * y
            # print(val)
            if val <= margin:
                for word, weight in w.items():
                    if abs(weight) < c:
                        w[word] = 0
                    else:
                        w[word] -= np.sign(weight) * c

                for word, count in phi.items():
                    w[word] += count * y

    text = ""
    for word, weight in w.items():
        text += f'{word}\t{weight}\n'
    # print(text)

    with open('train_answer_margin0.txt', 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
