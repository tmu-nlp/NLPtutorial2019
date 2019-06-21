from collections import defaultdict
import numpy as np


def train_svm(inpath: str, outpath: str):

    margin = 0.1
    c = 0.1

    with open(inpath, 'r', encoding='utf-8') as fin:

        # 重みの初期化
        w = defaultdict(float)

        for line in fin:
            y, x = line.split('\t')
            y = int(y)
            phi = create_features(x)

            # 内積の計算：dot
            val = calc_val(w, phi, y)

            if val <= margin:
                update_weights(w, phi, y, c)

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


def calc_val(w: dict, phi: dict, y: int) -> float:

    val = 0
    for name, value in phi.items():
        if name in w:
            val += value * w[name] * y
    
    return val


def update_weights(w: dict, phi: dict, y: int, c: float):
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= np.sign(value) * c
            
    for name, value in phi.items():
        w[name] += value * y
