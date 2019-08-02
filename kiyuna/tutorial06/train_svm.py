'''
L1 正則化とマージンで学習を行う train-svm
'''
import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def create_features(sentence):
    """ 素性関数をつくる """
    phi = defaultdict(int)
    for word in sentence.split():
        phi[f'UNI:{word}'] += 1
    return phi


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


def update_weights_L1reg(w, phi, y, c=0.0001):
    """ 更新のたびに、重みから定数 c を引く（L1 正則化）[スライド #06 p27, 29] """
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0     # 引きすぎないようにする
        else:
            w[name] -= sign(value) * c
    for name, value in phi.items():
        w[name] = w.get(name, 0) + value * y


def lazy_update_weights_L1reg(w, phi, y, last, iter, c=0.0001):
    """ 正則化は重みの使用時に行う """
    for name, value in phi.items():
        w[name] = getw(w, name, iter, last, c) + value * y


def getw(w, name, iter, last, c=0.0001):
    """
    L1 正則化 [スライド #06 p26]
    - 大小に関わらず同等の罰則
    - 多くの重みが 0 になるため小さなモデルが学習可能
    - c: 正則化係数（0.0001, 0.001, 0.01, 0.1, 1.0）

    L1 正則化を遅延評価する（lazy_update） [スライド #06 p30]
    - 正則化は重みの使用時に行う

    update_weights が呼ばれていない分も引いてしまっている気がする...
    """
    if iter != last[name]:
        c_size = c * (iter - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]


def dot(w, phi):
    score = 0
    for name, phi_val in phi.items():
        if name in w:
            score += w[name] * phi_val
    return score


def sign(val):
    return 1 if val >= 0 else -1


def predict_all(file_path, model_file, epoch=1, margin=0, c=0.0001, f=2):
    """ マージンを用いたオンライン学習 [スライド #06 p17-] """
    w = defaultdict(float)
    last = defaultdict(int)
    for e in range(epoch):
        message(f"[+] {e + 1:2d} / {epoch}", CR=True)
        for iter, line in enumerate(open(file_path)):
            y_label, x_sentence = line.split("\t")
            y_label = int(y_label)
            phi = create_features(x_sentence)
            '''
            # パーセプトロン
            #   - 各学習事例を分類してみる
            #   - 間違った答えを返す時に、重みを更新
            y' = sign(dot(w, φ))    <- margin = 0 としても一致しないのは sign のせい
            if y' != y
                update_weights(w, φ, y)
            # マージンを用いたオンライン学習 [スライド #06 p22]
            #   - 誤りだけでなく、一定のマージン以内の場合でも更新
            val = dot(w, φ) * y     <- 正しい分類結果は常に正
            if val <= margin
                update_weights(w, φ, y)
            '''

            # y_p = sign(dot(w, phi))
            # if y_p != y_label:
            #     update_weights(w, phi, y_label)

            val = dot(w, phi) * y_label
            if val <= margin:
                if f == 0:
                    update_weights(w, phi, y_label)
                elif f == 1:
                    update_weights_L1reg(w, phi, y_label, c)
                elif f == 2:
                    lazy_update_weights_L1reg(w, phi, y_label, last, iter, c)
        # 遅延評価が終了していないものがある気がする
    message()

    with open(model_file, 'w') as f:
        for key, val in sorted(w.items()):
            f.write(f"{key}\t{val}\n")


def load_model(model_file):
    w = defaultdict(float)
    for line in open(model_file):
        word, weight = line.split('\t')
        w[word] = float(weight)
    return w


def predict_one(w, phi):
    """ sign(dot(w, phi)) """
    score = sum(v * w[k] for k, v in phi.items())
    return sign(score)


def test_svm(model_file, input_file, output_file):
    w = load_model(model_file)
    with open(output_file, 'w') as f:
        for sentence in open(input_file):
            sentence = sentence.rstrip()
            phi = create_features(sentence)
            result = predict_one(w, phi)
            f.write(f"{result}\t{sentence}\n")


if __name__ == '__main__':
    train_file = '../../data/titles-en-train.labeled'
    test_file = '../../data/titles-en-test.word'
    model_file = './model_svm.txt'
    output_file = './result.labeled'

    with Timer() as t:
        predict_all(train_file, model_file,
                    epoch=2, margin=10, c=0.0001, f=1)
    print(f'lazy_update なし -> {t.secs:.1f} [s]')
    with Timer() as t:
        predict_all(train_file, model_file,
                    epoch=2, margin=10, c=0.0001, f=2)
    print(f'lazy_update あり -> {t.secs:.1f} [s]')
    exit()

    id = 2       # 0: L1regなし, 2: L1regあり
    epochs = [1, 5, 10, 15, 20, 25, 30]
    margins = [0, 1, 5, 10, 15, 20, 25, 30]
    E = np.zeros((len(margins), len(epochs)))
    for i, epoch in enumerate(epochs):
        for j, margin in enumerate(margins):
            print(f"[*] epoch = {epoch}, margin = {margin}")
            predict_all(train_file, model_file,
                        epoch=epoch, margin=margin, c=0.0001, f=id)
            test_svm(model_file, test_file, output_file)
            proc = subprocess.run(
                f'python2 ../../script/grade-prediction.py'
                f' ../../data/titles-en-test.labeled {output_file}'.split(),
                stdout=subprocess.PIPE
            )
            ans = proc.stdout.decode().rstrip().split()[-1].replace("%", "")
            print(ans)
            E[j][i] = float(ans)

    # 以下，result.png
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mappable = ax.pcolor(E, cmap='jet', edgecolors='k', alpha=0.8)
    fig.colorbar(mappable)

    ma_y, ma_x = np.where(E == E.max())
    ax.scatter(ma_x + 0.5, ma_y + 0.5, c='r', label='max')
    mi_y, mi_x = np.where(E == E.min())
    ax.scatter(mi_x + 0.5, mi_y + 0.5, c='b', label='min')
    print("[+] max:", E.max(), np.where(E == E.max()))
    print("[+] min:", E.min(), np.where(E == E.min()))

    ax.set_xticks(np.arange(len(epochs)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(margins)) + 0.5, minor=False)
    ax.set_xticklabels(epochs, minor=False)
    ax.set_yticklabels(margins, minor=False)
    ax.set_xlabel('$epoch$')
    ax.set_ylabel('$margin$')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.set_title('c = 0.0001 (L1reg {})'.format("なし" if id == 0 else "あり"))
    plt.savefig('L1reg{}.png'.format("なし" if id == 0 else "あり"))

    message("[+] Done.")


'''
Accuracy = 93.800921% (L1正則化あり, epoch=15, margin=30)
Accuracy = 93.765498% (L1正則化なし, epoch=20, margin=30)
Accuracy = 90.967056% (#05 の perceptron と一致, mergin = 0 にするなど)
'''
