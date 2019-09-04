from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle
from sys import argv
from os import makedirs
from time import time


start = time()


class NN():
    def __init__(self, hidden_size, layer_num):
        self.H = hidden_size
        self.L = layer_num
        self.ids = defaultdict(lambda: len(self.ids))

    def create_features(self, sentence):
        phi = np.zeros((len(self.ids)))
        words = sentence.split()
        for word in words:
            if f'UNI:{word}' not in self.ids:
                continue
            phi[self.ids[f"UNI:{word}"]] += 1
        return phi

    def foward_nn(self, φ0):
        φ = [φ0]  # 各層の値
        for i in range(len(self.net)):
            w, b = self.net[i]
            # 前の層に基づいて値を計算
            φ.append(np.tanh(np.dot(w, φ[i]) + b))
        return φ  # 各層の結果を返す

    def backward_nn(self, φ, y_):
        J = len(self.net)
        δ = [np.ndarray for _ in range(J)]
        δ.append(y_ - φ[J])
        δ_ = np.zeros(J + 1, dtype=np.ndarray)
        for i in range(J-1, -1, -1):
            δ_[i + 1] = δ[i + 1] * (1 - φ[i + 1] ** 2)
            w, b = self.net[i]
            δ[i] = np.dot(δ_[i + 1], w)
        return δ_

    def update_weights(self, φ, δ_, λ):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += λ * np.outer(δ_[i + 1], φ[i])
            b += λ * δ_[i + 1]

    def init_net(self):
        net = []
        np.random.seed(42)
        w_0 = np.random.rand(self.H, len(self.ids)) / 5 - 0.1
        b_0 = np.random.rand(self.H) / 5 - 0.1
        net.append((w_0, b_0))

        # while len(net) < self.L:
        for _ in range(self.L - 1):
            w = np.random.rand(self.H, self.H) / 5 - 0.1
            b = np.random.rand(self.H) / 5 - 0.1
            net.append((w, b))

        w_o = np.random.rand(1, self.H) / 5 - 0.1
        b_o = np.random.rand(1) / 5 - 0.1
        net.append((w_o, b_o))
        self.net = net

    def prepare_training(self, train_file: str):
        data = []
        for line in open(train_file, "r"):
            label, sentence = line.rstrip().split('\t')
            data.append((int(label), sentence))
            for word in sentence.split():
                self.ids[f"UNI:{word}"]
        self.feat_lab = [(self.create_features(sent), lab) for lab, sent in data]

    def train(self, in_path, epoch):
        λ = 0.1
        # ファイルを読み込み、素性を作る
        self.prepare_training(in_path)
        self.ids = dict(self.ids)
        self.init_net()
        for _ in tqdm(range(epoch)):
            for φ0, y in tqdm(self.feat_lab):
                φ = self.foward_nn(φ0)
                δ_ = self.backward_nn(φ, y)
                self.update_weights(φ, δ_, λ)

    def dump(self):
        makedirs('pickles', exist_ok=True)
        with open(f"./pickles/net.pkl", 'wb') as f_out:
            pickle.dump(self.net, f_out)
        with open(f"./pickles/ids.pkl", 'wb') as f_out:
            pickle.dump(self.ids, f_out)

    def load(self):
        with open(f"./pickles/net.pkl", 'rb') as f_in:
            self.net = pickle.load(f_in)
        with open(f"./pickles/ids.pkl", 'rb') as f_in:
            self.ids = pickle.load(f_in)

    def predict_one(self, sentence):
        φ = self.create_features(sentence)
        for i in range(len(self.net)):
            w, b = self.net[i]
            φ = ((np.tanh(np.dot(w, φ) + b)))
        return 1 if φ >= 0 else -1

    def test(self, in_path, out_path):
        with open(out_path, 'w') as f_out:
            for line in open(in_path, "r"):
                sentence = line.rstrip()
                predict = self.predict_one(sentence)
                print(f'{predict}\t{sentence}', file=f_out)


if __name__ == "__main__":
    nn = NN(int(argv[1]), int(argv[2]))     # hidden_size, layer_num
    # nn.train("../../../nlptutorial/test/03-train-input_5.txt", 1000)
    # nn.test("../../../nlptutorial/test/03-train-input_5.txt")

    if argv[4:] != ["test"]:
        nn.train("../../../nlptutorial/data/titles-en-train.labeled", int(argv[3]))
        nn.dump()
    else:
        nn.load()
    nn.test("../../../nlptutorial/data/titles-en-test.word", './out.txt')    
    end = time()
    print(f"elapsed time = {end - start} s")
