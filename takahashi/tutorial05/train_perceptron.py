from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

Map = Dict[str, float]

# uni-gram の素性作成
def create_features(words: List[str]) -> Map:
    phi = defaultdict(float)  # type: Map

    for word in words:
        phi["UNI:" + word] += 1
    return phi


# 1 つの事例に対する予測
def predict_one(w: Map, phi: Map) -> int:
    score = 0
    for name, val in phi.items():
        if name in w:
            score += val * w[name]
    if score >= 0:
        return 1
    else:
        return -1

# 重みの更新
def update_weights(w, phi, y):
    for name, val in phi.items():
        w[name] += val * y


def train(file_path: str) -> Map:
    w = defaultdict(float)  # type: Map
    epoch = 20
    for _ in tqdm(range(epoch)):
        for line in open(file_path, "r", encoding="utf-8"):
            line = line.strip().split()
            label, words = int(line[0]), line[1:]
            phi = create_features(words)
            pre = predict_one(w, phi)
            if pre != label:
                update_weights(w, phi, label)
    return w


if __name__ == "__main__":
    train_file = "../files/data/titles-en-train.labeled"
    output = "model_file"

    with open(output, "w") as f:
        for (key, val) in sorted(train(train_file).items(), key=lambda x: x[0]):
            f.write(f"{key}\t{val:.6f}" + "\n")

"""
  - epoch = 20, no preprocess
    Accuracy = 93.871768%

  - epoch = 20, remove_stopwords

"""
