from typing import Dict, Tuple
from collections import defaultdict
from math import log2
import sys

def load_model() -> Dict[str, float]:
    probabilities = defaultdict(int)
    for line in open("model_file.txt", "r"):
        word, p = line.rstrip("\n").split("\t")
        probabilities[word] = float(p)
    return probabilities

def test_unigram(test_file: str) -> Tuple[float, float]:
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    v = 1e6
    w = 0
    h = 0
    unk = 0

    # モデルの読み込み
    probabilities = load_model()

    # 評価
    for line in open(test_file, "r"):
        words = line.rstrip("\n").split(" ")
        words.append("</s>")
        for word in words:
            w += 1
            p = lambda_unk / v
            if word in probabilities.keys():
                p += lambda_1 * probabilities[word]
            else:
                unk += 1
            h += -log2(p)

    entropy = h / w
    coverage = (w - unk) / w
    return entropy, coverage

if __name__ == "__main__":
    args = sys.argv
    entropy, coverage = test_unigram(args[1])
    print(f"entropy = {entropy:.6f}")
    print(f"coverage = {coverage:.6f}")
