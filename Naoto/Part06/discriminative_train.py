import os
import sys
import subprocess
from collections import defaultdict
from math import exp, copysign


# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def create_features(x):
    phi = defaultdict(lambda: 0)
    for word in x:
        phi["UNI:" + word] += 1
    return phi


def getw(w, name, c, iter_, last):
    if iter_ != last[name]:
        c_size = c * (iter_ - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= copysign(1, w[name]) * c_size
        last[name] = iter_
    return w[name]


def predict_one(w, phi: {}, iter_, last, c):
    score = 0
    for name, value in phi.items():
        if name in w:
            # getw(w, name, c, iter_, last)
            score += value * w[name]
    return score, exp(score) / (1 + exp(score))


def update_weights(w, phi, score, y, c, α):
    for name, value in w.items():
        if abs(value) <= c:
            w[name] = 0
        else:
            w[name] -= copysign(1, w[name]) * c
    for name, value in phi.items():
        w[name] += α * value * y * exp(score)/((1 + exp(score))**2)


def train(iput_, model, iterations, c):
    α = 1
    margin = 0
    last = defaultdict(lambda: 0)
    with open(input_) as f, open(model + str(c) + ".txt", "w") as fw:
        w = {}
        for iter_ in range(int(iterations)):
            # if iter != 0:
            #     α *= 0.85
            for line in f:
                x_y = line.split()
                y = int(x_y[0])
                x = x_y[1:]
                phi = create_features(x)
                for k in phi.keys():
                    if not(k in w):
                        w[k] = 0
                score, y_ = predict_one(w, phi, iter_, last, c)
                update_weights(w, phi, score, y, c, α)
                val = score * y_
                if val <= margin:
                    update_weights(w, phi, score, y, c, α)
        for k, v in sorted(w.items()):
            print(f"{k} {v:.6f}", file=fw)


if __name__ == "__main__":
    input_ = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/data/titles-en-train.labeled"
    model = "./model"
    c_1 = 10000
    c = []
    for i in range(10):
        if i % 2 == 0:
            c.append(1/c_1)
        else:
            c.append(3/c_1)
            c_1 /= 10
    for i, c_value in enumerate(c):
        if i >= 0 and i <= 10:
            train(input_, model, 1, c_value)
    message("[+] Finished!")
