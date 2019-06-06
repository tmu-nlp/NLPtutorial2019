'''
2-gram モデルを学習
'''
import os
import sys
import subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text):
    print("\33[92m" + text + "\33[0m")


def n_gram(seq, n):
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def count_words(path):
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    with open(path) as f:
        for line in f:
            words = ["<s>"] + line.rstrip().split() + ["</s>"]
            for token in n_gram(words, 2):  # w_{i-1}, w_i
                # 2-gram（分子, 分母）
                counts[' '.join(token)] += 1
                context_counts[' '.join(token[:-1])] += 1
                # 1-gram（分子, 分母）
                counts[token[-1]] += 1
                context_counts[""] += 1
    return counts, context_counts


def train_bigram(train, model):
    counts, context_counts = count_words(train)
    with open(model, 'w') as f:
        for ngram, count in sorted(counts.items()):  # str, tuple の混在はダメ
            words = ngram.split()
            context = ' '.join(words[:-1])
            probability = counts[ngram] / context_counts[context]
            f.write(f'{ngram}\t{probability:f}\n')


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        train = '../../test/02-train-input.txt'
        model = './model_test.txt'
    else:
        message("[*] wiki")
        train = '../../data/wiki-en-train.word'
        model = './model_wiki.txt'

    train_bigram(train, model)
    if is_test:
        subprocess.run(
            f'diff -s {model} ../../test/02-train-answer.txt'.split())
    message("[+] Finished!")
