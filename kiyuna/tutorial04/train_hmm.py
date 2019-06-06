'''
隠れマルコフモデルによる品詞推定
'''
import os
import sys
import subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text):
    print("\33[92m" + text + "\33[0m")


def train_hmm(train, model):
    context = defaultdict(int)
    transition = defaultdict(int)
    emit = defaultdict(int)
    with open(train) as f:
        for line in map(lambda x: x.strip(), f):
            previous = "<s>"
            context[previous] += 1
            for wordtags in line.split():
                word, tag = wordtags.split("_")
                transition[previous + " " + tag] += 1   # 遷移を数え上げる
                context[tag] += 1                       # 文脈を数え上げる
                emit[tag + " " + word] += 1             # 生成を数え上げる
                previous = tag
            transition[previous + " </s>"] += 1
    with open(model, 'w') as f:
        # 遷移確率を出力
        for key, value in sorted(transition.items()):
            previous, word = key.split()
            f.write(f'T {key} {value / context[previous]:f}\n')
        # 生成確率を出力
        for key, value in sorted(emit.items()):
            previous, word = key.split()
            f.write(f'E {key} {value / context[previous]:f}\n')


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        train = '../../test/05-train-input.txt'
        model = './model_test.txt'
    else:
        message("[*] wiki")
        train = '../../data/wiki-en-train.norm_pos'
        model = './model_wiki.txt'

    train_hmm(train, model)
    if is_test:
        subprocess.run(
            f'diff -s {model} ../../test/05-train-answer.txt'.split()
        )
    message("[+] Done!")
