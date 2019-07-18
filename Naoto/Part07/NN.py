import numpy as np
import dill
from collections import defaultdict
from tqdm import tqdm
from typing import List

# type hinting
array = np.ndarray


class TrainNN:
    def __init__(self, layer: int, node: int, epoch: int):
        # hyperparameter
        self.L = layer
        self.N = node
        self.epoch = epoch

        self.ids = defaultdict(lambda: len(self.ids))
        self.net = []
        self.feat_lab = []

    def forward(self, phi0: array):
        phis = [phi0]

        for i in range(len(self.net)):
            weight, bias = self.net[i]
            phis.append(np.tanh(np.dot(weight, phis[i]) + bias))
        return phis

    def backward(self, phis: List[array], label: int):
        J = len(self.net)
        delta = [np.ndarray for _ in range(J)]
        delta.append(label - phis[J])
        dd = [0 for _ in range(J + 1)]
        for i in range(0, J)[::-1]:
            dd[i + 1] = delta[i + 1] * (1 - phis[i + 1] ** 2)
            weight, _ = self.net[i]
            delta[i] = np.dot(dd[i + 1], weight)
        return dd

    def update_weights(self, phis, dd, lam=0.1):
        for i in range(len(self.net)):
            weight, bias = self.net[i]
            weight += lam * np.outer(dd[i + 1], phis[i])
            bias += lam * dd[i + 1]     

    def create_features(self, sentence: str):
        # uni-gramで素性作成
        phi = [0 for _ in range(len(self.ids))]
        for word in sentence.split():
            phi[self.ids[f"UNI:{word}"]] += 1
        return phi

    def prepare_training(self, train_file: str):
        data = []
        for line in open(train_file, "r", encoding="utf-8"):
            label, sentence = line.rstrip().split("\t")
            data.append((int(label), sentence))
            for word in sentence.split():
                self.ids[f"UNI:{word}"]
            self.feat_lab = [(self.create_features(sent), lab) for lab, sent in data]

    def initialize_network(self):
        # np.random.rand(n, m) : 0 ~ 1 未満の n 行 m 列の一様乱数のリストを生成

        # 入力層
        w_in = np.random.rand(self.N, len(self.ids)) / 5 - 0.1
        b_in = np.random.rand(self.N) / 5 - 0.1
        self.net.append((w_in, b_in))

        # 隠れ層
        for _ in range(self.L - 1):
            w = np.random.rand(self.N, self.N) / 5 - 0.1
            b = np.random.rand(self.N) / 5 - 0.1
            self.net.append((w, b))

        # 出力層
        w_out = np.random.rand(1, self.N) / 5 - 0.1
        b_out = np.random.rand(1) / 5 - 0.1
        self.net.append((w_out, b_out))

    def train(self):
        # ファイルを読み込み、素性を作る
        self.prepare_training("/Users/naoto_nakazawa/komachi_lab/nlptutorial/test/03-train-input.txt")
        # ネットワークを一様乱数で初期化
        self.initialize_network()

        # train
        for _ in range(self.epoch):
            for feat0, label in tqdm(self.feat_lab):
                feat = self.forward(feat0)
                delta_ = self.backward(feat, label)
                self.update_weights(feat, delta_)

        return self.net, self.ids


class TestNN():
    def __init__(self, net, ids):
        self.net = net
        self.ids = ids

    def create_feature(self, sentence: str):
        phi = [0] * len(ids)
        words = sentence.split()
        for word in words:
            key = f"UNI:{word}"
            if key in self.ids:
                phi[self.ids[key]] += 1
        return phi

    def predict_one(self, phi0):
        phis = [phi0]
        for i in range(len(self.net)):
            weight, bias = self.net[i]
            phis.append(np.tanh(np.dot(weight, phis[i]) + bias))
        return phis[len(self.net)][0]

    def test(self):
        test_file = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/test/03-train-input.txt"
        ans = []
        for sentence in open(test_file, "r", encoding="utf-8"):
            sentence = sentence.rstrip()
            phi0 = self.create_feature(sentence)
            score = self.predict_one(phi0)
            ans.append("1" if score > 0 else "-1")

        with open("answer", "w", encoding="utf-8") as out:
            out.write("\n".join(ans))


if __name__ == "__main__":
    # train
    train_nn = TrainNN(1, 2, 20)
    network, ids = train_nn.train()

    # test
    test_nn = TestNN(network, ids)
    test_nn.test()
