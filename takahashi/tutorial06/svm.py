from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
import numpy as np
Map = Dict[str, float]


# 素性の作成
def create_features(sentence: str) -> Map:
    words = sentence.split()
    phi = defaultdict(float)  # type: Map
    for word in words:
        phi[f"UNI:{word}"] += 1
    return phi


# 重みの更新
def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


# 1 つの事例に対する予測 (L1 正則化を使う)
def single_example(w: Map, phi: Map, i: int, last: Map) -> int:
    score = 0
    for name, val in phi.items():
        if name in w:
            score += val * L1_regularize(w, phi, name, i, last)
    return score


def sign(val):
    return 1 if val >= 0 else -1


# L1 正則化を遅延評価する
def L1_regularize(w: Map, phi: Map, name: str, iter: int, last: Map, c=0.0001) -> float:
    if iter != last[name]:
        c_size = c * (iter - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]


# オンライン学習
def train(file_path: str, mergin=20):
    weights = defaultdict(float)  # type: Map
    last = defaultdict(int)
    epoch = 10
    for _ in tqdm(range(epoch)):
        for i, line in enumerate(open(file_path, "r", encoding="utf-8")):
            str_label, sentence = line.rstrip().split("\t")
            label = int(str_label)
            phi = create_features(sentence)
            val = label * single_example(weights, phi, i, last)
            if val <= mergin:
                update_weights(weights, phi, label)

    with open("model", "w") as out:
        for (key, val) in sorted(weights.items()):
            out.write(f"{key}\t{val}\n")


def load_model(model_file: str) -> Map:
    w = defaultdict(float)  # type: Map
    for line in open(model_file, "r", encoding="utf-8"):
        word, weight = line.strip().split("\t")
        w[word] = float(weight)
    return w


def predict_one(w: Map, phi: Map) -> int:
    score = sum(v * w[k] for k,v in phi.items())
    return 1 if score >= 0 else -1


def test(model_file: str, input_file: str):
    w = load_model(model_file)
    with open("answer", "w") as f:
        for line in open(input_file):
            line = line.rstrip()
            phi = create_features(line)
            pre = predict_one(w, phi)
            f.write(f"{pre}\t{line}\n")


if __name__ == "__main__":
    train_file = "../files/data/titles-en-train.labeled"
    test_file = "../files/data/titles-en-test.word"
    model_file = "./model"

    train(train_file)
    test(model_file, test_file)

"""
$ ../files/script/grade-prediction.py ../files/data/titles-en-test.labeled answer
Accuracy = 93.552958% (svm)

Accuracy = 93.234148% (perceptron)
"""