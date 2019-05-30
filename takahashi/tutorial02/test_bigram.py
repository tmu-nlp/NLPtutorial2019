import sys
from math import log2
from typing import Dict
from collections import defaultdict
from train_bigram import get_ngram
import numpy as np

Map = Dict[str, float]


def load_model(model_file="./model_file.txt") -> Map:
    probs = defaultdict(float)  # type: Map
    for line in open(model_file, "r"):
        ws, p = line.strip("\n").split("\t")
        probs[ws] = float(p)
    return probs


def test_bigram(test_file: str) -> None:
    # モデルの読み込み
    probs = load_model()

    # 評価
    lambda_range = np.arange(0.05, 1.0, 0.05)
    for lambda_1 in lambda_range:
        for lambda_2 in lambda_range:
            V = 1e6
            W = 0
            H = 0
            for line in open(test_file, "r"):
                words = line.strip().split(" ")
                W += len(words) + 2  # 文頭/文末記号分
                for ws in get_ngram(["<s>", *words, "</s>"], 2):
                    p1 = lambda_1 * probs[ws[1]] + (1 - lambda_1) / V
                    p2 = lambda_2 * probs[" ".join(ws)] + (1 - lambda_2) * p1
                    H += -log2(p2)
            print(f"{lambda_1:.2f} {lambda_2:.2f}\t\t{H/W:.6f}")


if __name__ == "__main__":
    args = sys.argv
    test_bigram(args[1])
