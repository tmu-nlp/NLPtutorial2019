import os
import sys
import subprocess
from collections import defaultdict


# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def create_features(x):
    phi = defaultdict(lambda: 0)
    for word in x:
        phi["UNI:" + word] += 1
    return phi


def predict_one(w, phi: {}):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1, score
    else:
        return -1, score


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


def train(iput_, model, iterations):
    with open(input_) as f, open(model, "w") as fw:
        w = {}
        for _ in range(int(iterations)):
            for line in f:
                x_y = line.split()
                y = int(x_y[0])
                x = x_y[1:]
                phi = create_features(x)
                for k in phi.keys():
                    if not(k in w):
                        w[k] = 0
                y_ = predict_one(w, phi)
                if y_ != y:
                    update_weights(w, phi, y)
        for k, v in sorted(w.items()):
            print(f"{k} {v:.6f}", file=fw)


if __name__ == "__main__":
    is_test = sys.argv[1] == "test"

    if is_test:
        message("[*] test")
        input_ = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/test/03-train-input.txt"
        model = "./05-model.txt"
    else:
        message("[*] wiki")
        input_ = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/data/titles-en-train.labeled"
        model = "./model_wiki.txt"

    train(input_, model, sys.argv[2])

    if is_test:
        ans = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/test/03-train-answer.txt"
        subprocess.run(f"diff -s {model} {ans}".split())

    message("[+] Finished!")
