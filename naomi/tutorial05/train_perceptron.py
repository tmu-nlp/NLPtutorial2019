from collections import defaultdict
# import re


def train_perceptron(inpath: str, outpath: str):

    with open(inpath, 'r', encoding='utf-8') as fin:

        # 重みの初期化
        w = defaultdict(float)

        for line in fin:
            y, x = line.split('\t')
            y = int(y)
            phi = create_features(x)
            y_ = predict_one(w, phi)

            if y_ != y:
                update_weights(w, phi, y)

    with open(outpath, 'w+', encoding='utf-8') as fout:
        for name, value in sorted(w.items()):
            print('{0}\t{1:6f}'.format(name, value), file=fout)


def create_features(x: str) -> dict:

    phi = defaultdict(int)

    # []の中の文字集合, \w: Unicode単語文字, +: 直前のREを１回以上繰り返す
    # ptn = r'''[\w']+'''
    # words = re.findall(ptn, x)
    words = x.split()

    for word in words:
        # UNI: を追加して1-gramを表す
        phi['UNI:' + word] += 1

    return phi


def predict_one(w: dict, phi: dict) -> int:

    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]

    if score >= 0:
        return 1
    else:
        return -1


def update_weights(w: dict, phi: dict, y: int):
    for name, value in phi.items():
        w[name] += value * y
