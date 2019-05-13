import sys
from math import log2
from typing import Dict
from collections import defaultdict
from train_bigram import get_ngram

Map = Dict[str, float]


def load_model(model_file="./model_file.txt") -> Map:
    probs = defaultdict(float)  # type: Map
    for line in open(model_file, "r"):
        ws, p = line.strip("\n").split("\t")
        probs[ws] = float(p)
    return probs


def test_bigram(test_file: str) -> None:
    V = 1e6
    W = 0
    H = float(0)
    lambda_1 = 0.95
    lambda_2 = 0.95

    # モデルの読み込み
    probs = load_model()

    # 評価
    for line in open(test_file, "r"):
        words = line.strip().split(" ")
        W += len(words) + 2  # 文頭/文末記号分
        for ws in get_ngram(["<s>", *words, "</s>"], 2):
            p1 = lambda_1 * probs[ws[1]] + (1 - lambda_1) / V
            p2 = lambda_2 * probs[" ".join(ws)] + (1 - lambda_2) * p1
            H += -log2(p2)
    print(f"entropy = {H/W:.6f}")


if __name__ == "__main__":
    args = sys.argv
    test_bigram(args[1])
