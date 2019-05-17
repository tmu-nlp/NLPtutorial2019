'''
2-gram モデルに基づいて評価データのエントロピーを計算

# TODO: Witten-Bell 平滑化を利用, 任意な文脈長が利用可能なプログラム
'''
import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Sequence

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def n_gram(seq: Sequence, n: int) -> tuple:
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def load_model(path):
    probs = defaultdict(float)
    with open(path) as f:
        for line in f:
            token, prob = line.split('\t')
            probs[token] = float(prob)
    return probs


def test_bigram(probs, test, λ_1=0.95, λ_2=0.95):
    # 線形補間
    # P(w_i ∣ w_{i−1}) = λ2 P_ML(w_i ∣ w_{i−1}) + (1 − λ_2) P(w_i)
    # P(w_i) = λ_1 P_ML(w_i) + (1 − λ_1) / N
    V = 1000000         # 未知語を含む語彙数
    W = 0               # 単語数
    H = 0               # 負の底 2 の対数尤度
    with open(test) as f:
        for line in f:
            words = ["<s>"] + line.rstrip().split() + ["</s>"]
            for token in n_gram(words, 2):  # w_{i-1}, w_i
                P1 = λ_1 * probs[token[-1]] + (1 - λ_1) / V
                P2 = λ_2 * probs[' '.join(token)] + (1 - λ_2) * P1
                H += -math.log2(P2)
                W += 1
    entropy = H / W
    return entropy


if __name__ == '__main__':
    message("[*] wiki")
    path = './model_wiki.txt'
    test = '../../data/wiki-en-test.word'

    model = load_model(path)

    # グリッド探索
    message("[*] Grid Search")
    E = np.zeros((19, 19))
    for j, λ_2 in enumerate(np.arange(5, 100, 5)):
        message(f"[+] {j + 1:2d} / 19", CR=True)
        for i, λ_1 in enumerate(np.arange(5, 100, 5)):
            E[j, i] = test_bigram(model, test, λ_1 / 100, λ_2 / 100)
    message()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mappable = ax.pcolor(E, cmap='jet', edgecolors='k', alpha=0.8)
    fig.colorbar(mappable)
    ax.set_xticks(np.arange(19) + 0.5, minor=False)
    ax.set_yticks(np.arange(19) + 0.5, minor=False)
    ma_y, ma_x = np.where(E == E.max())
    ax.scatter(ma_x + 0.5, ma_y + 0.5, c='r', label='max')
    mi_y, mi_x = np.where(E == E.min())
    ax.scatter(mi_x + 0.5, mi_y + 0.5, c='b', label='min')
    ax.set_xticklabels(np.arange(5, 100, 5), minor=False)
    ax.set_yticklabels(np.arange(5, 100, 5), minor=False)
    ax.set_xlabel(r'$λ_1$')
    ax.set_ylabel(r'$λ_2$')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.savefig('result.png')

    message("[+] Finished!")
