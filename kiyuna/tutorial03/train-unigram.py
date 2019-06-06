'''
文字 1-gram モデルを学習
'''
import os
import sys
import subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text):
    print("\33[92m" + text + "\33[0m")


def count_words(path):
    cnter = defaultdict(int)
    total_cnt = 0
    with open(path) as f:
        for line in f:
            words = line.rstrip().split() + ["</s>"]     # append EOS
            for w in words:
                cnter[w] += 1
                total_cnt += 1
    return cnter, total_cnt


def train_unigram(train, model):
    cnter, total = count_words(train)
    with open(model, 'w') as f:
        for k, v in sorted(cnter.items()):
            f.write(f'{k}\t{v / total:f}\n')


if __name__ == '__main__':
    message("[*] wiki")
    train = '../../data/wiki-ja-train.word'
    model = './model_wiki.txt'

    train_unigram(train, model)
    message("[+] Finished!")
