'''
パーセプトロンを用いた分類器学習

人物なら 1, そうでなければ -1
'''
import os
import sys
import subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def create_features(sentence):
    phi = defaultdict(int)
    for word in sentence.split():
        phi[f'UNI:{word}'] += 1
    return phi


def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


def train_perceptron(train_file, model_file, epoch=1):
    w = defaultdict(float)
    for e in range(epoch):
        message(f"[+] {e + 1:3d} / {epoch}", CR=True)
        with open(train_file) as f:
            for line in f:
                y_label, x_sentence = line.split('\t')
                y_label = int(y_label)
                phi = create_features(x_sentence)
                y_predicted = predict_one(w, phi)
                if y_predicted != y_label:
                    update_weights(w, phi, y_label)
    message()
    with open(model_file, 'w') as f:
        for k, v in sorted(w.items()):
            print(f'{k}\t{v:f}', file=f)


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        train = '../../test/03-train-input.txt'
        model = './model_test.txt'
    else:
        message("[*] wiki")
        train = '../../data/titles-en-train.labeled'
        model = './model_wiki.txt'

    train_perceptron(train, model, epoch=1)

    if is_test:
        subprocess.run(
            f'diff -s {model} ../../test/03-train-answer.txt'.split()
        )

    message("[+] Done!")
