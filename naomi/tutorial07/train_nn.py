from collections import defaultdict
import numpy as np
import pickle


ids = defaultdict(lambda: len(ids))


def train_nn(inpath: str, wpath: str):

    feat_lab = []
    N = 2  # ノード
    L = 1  # レイヤ
    epoch = 1

    feat_lab = []

    with open(inpath, 'r', encoding='utf-8') as fin:

        for line in fin:
            y, x = line.split('\t')
            for word in x.split():
                ids['UNI:' + word]
        
        for _ in range(epoch):
            for line in fin:
                y, x = line.split('\t')
                y = int(y)
                feat_lab.append((create_features(x), y))

        net = init_network(len(ids), N, L)

        # 学習を行う
        for (phi_0, y) in feat_lab:
            phi = forward_nn(net, phi_0)
            delta = backward_nn(net, phi, y)
            update_weights(net, phi, delta, l)

    with open(wpath, 'wb') as f:
        pickle.dump(net, f)
        pickle.dump(dict(ids), f)


def create_features(x: str) -> dict:

    phi = [0 for _ in range(len(ids))]

    for word in x.split():
        # UNI: を追加して1-gramを表す
        phi[ids['UNI:' + word]] += 1

    return phi


def forward_nn(net: list, phi_0: np.ndarray) -> list:

    # 各層の値
    phi = [phi_0]

    for i in range(len(net)):
        # この層の重み、バイアス
        (w, b) = net[i]

        # 前の層の値に基づいて値を計算
        phi.append(np.tanh(np.dot(w, phi[i-1]) + b))

    return phi


def backward_nn(net: list, phi: list, y: int):
    J = len(net)

    delta = np.zeros(J+1)
    delta[-1] = np.array(y-phi[J][0])
    delta_ = np.zeros(J+1)

    for i in range(J-1, 0, -1):
        delta_[i+1] = delta[i+1]*(1-phi[i+1]**2)
        (w, b) = net[i]
        delta[i] = np.dot(delta_[i+1], w)
    return delta_


def init_network(feature_size, node, layer) -> list:
    # Networkの初期化（ランダム）
    # np.random.rand(n, m): 0.0以上, 1.0未満のｍ要素のリストｘnのリスト

    net =[]

    # 1つ目の隠れ層
    w0 = 2 * np.random.rand(node, feature_size) -1 # 重みは-1.0以上1.0未満で初期化
    b0 = np.random.rand(1, node)
    net.append((w0, b0))

    # 中間層
    while len(net) < layer:
        w = 2 * np.random.rand(node, node) - 1
        b = np.random.rand(1, node)
        net.append((w, b))
    
    # 出力層
    w_o = 2 * np.random.rand(1, node) - 1
    b_o = np.random.rand(1, 1)

    net.append((w_o, b_o))

    return net


def update_weights(net, phi, delta_, l):
    for i in range(len(net)):
        w, b = net[i]
        w += l * np.outer(delta_[i+1], phi[i])
        b += l * delta_[i+1]


def main():
    # 学習データ
    ftrain = '../../test/03-train-input.txt'
    ftrain = '../../data/titles-en-train.labeled'

    fnet = 'net_ibs'

    # 学習データのリストを読み込む
    train_nn(ftrain, fnet)


if __name__ == '__main__':
    main()
