import numpy as np
from collections import defaultdict
import joblib
from tqdm import tqdm
# 学習1回、隠れ層1つ、隠れ層のノード2つ

# train-nn.py
class NeuralNet:
    def __init__(self, layer_num, node_num):
        self.L = layer_num
        self.N = node_num
        self.ids = defaultdict(lambda: len(self.ids))
        self.feat_lab = []
        self.net = []

    def prepare(self, input_file):
        input_data = []
        with open(input_file) as f:
            for line in f:
                label, sentence = line.rstrip().split("\t")
                for word in sentence.split():
                    self.ids["UNI:" + word]
                input_data.append([sentence, int(label)])
        for sentence, label in input_data:
            self.feat_lab.append([self.create_features(sentence), label])

    def init_net(self):
        w_in = np.random.rand(self.N, len(self.ids)) / 5 - 0.1
        b_in = np.random.rand(self.N) / 5 - 0.1
        self.net.append((w_in, b_in))

        for _ in range(self.L - 1):
            w = np.random.rand(self.N, self.N) / 5 - 0.1
            b = np.random.rand(self.N) / 5 - 0.1
            self.net.append((w, b))

        w_out = np.random.rand(1, self.N) / 5 - 0.1
        b_out = np.random.rand(1) / 5 - 0.1
        self.net.append((w_out, b_out))

    def train(self, num_iter=10):
        for i in tqdm(range(num_iter)):
            for phi_0, y in tqdm(self.feat_lab):
                phi = self.forward_nn(phi_0)
                delta_d = self.backward_nn(phi, y)
                self.update_weights(phi, delta_d, 0.1)
        return self.net, self.ids
        # dill だるい
        # joblib.dump(self.net, "weight_file")
        # joblib.dump(self.ids, "id_file")

    def test(self, sentence):
        phi0 = self.create_features(sentence.rstrip(), False)
        score = self.predict_one(phi0)
        return 1 if score > 0 else -1

    def predict_one(self, phi0):
        phis = [phi0]
        for i in range(len(self.net)):
            w, b = self.net[i]
            phis.append(np.tanh(np.dot(w, phis[i]) + b))
        return phis[len(self.net)][0]

    # def load(self, weight_file, id_file):
    #     self.net = joblib.load(weight_file)
    #     self.ids = joblib.load(id_file)

    def create_features(self, sentence, is_train=True):
        phi = [0 for _ in range(len(self.ids))]
        if is_train:
            for word in sentence.split():
                key = "UNI:" + word
                phi[self.ids[key]] += 1
        else:
            for word in sentence.split():
                key = "UNI:" + word
                if key in self.ids:
                    phi[self.ids[key]] += 1
        return phi

    def forward_nn(self, phi_0):
        phi = [0 for _ in range(len(self.net)+1)]
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
        return phi

    def backward_nn(self, phi, y_d):
        J = len(self.net)
        delta = [0 for _ in range(J)]
        delta.append(y_d - phi[J])
        delta_d = [0 for _ in range(J+1)]
        for i in reversed(range(J)):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
            w, b = self.net[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    def update_weights(self, phi, delta_d, lamb):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lamb * np.outer(delta_d[i+1], phi[i])
            b += lamb * delta_d[i+1]

if __name__ == "__main__":
    input_path = "../../data/titles-en-train.labeled"
    test_path = "../../data/titles-en-test.word"
    output_path = "ans"
    # input_path = "titles-en-train.labeled"
    net = NeuralNet(1, 2)
    print("Loading data...")
    net.prepare(input_path)
    net.init_net()
    print("Training...")
    net.train(1)
    print("Predicting...")
    ans = []
    for sentence in open(test_path):
        pred = net.test(sentence)
        ans.append(str(pred))
    with open(output_path, mode="w") as fw:
        fw.write("\n".join(ans))
    print("Finished!")




