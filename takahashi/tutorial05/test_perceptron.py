from collections import defaultdict
from typing import Dict, List

from train_perceptron import create_features, predict_one

Map = Dict[str, float]


def load_model(model_file: str) -> Map:
    w = defaultdict(float)  # type: Map
    for line in open(model_file, "r", encoding="utf-8"):
        word, weight = line.strip().split("\t")
        w[word] = float(weight)
    return w


def predict_all(model_file: str, input_file: str) -> None:
    w = load_model(model_file)
    with open("answer", "w") as f:
        for line in open(input_file):
            line = line.rstrip()
            phi = create_features(line.split())
            pre = predict_one(w, phi)
            f.write(f"{pre}\t{line}" + "\n")


if __name__ == "__main__":
    test_file = "../files/data/titles-en-test.word"
    model_file = "./model_file"
    predict_all(model_file, test_file)
