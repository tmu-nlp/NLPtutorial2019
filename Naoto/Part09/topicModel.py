from collections import defaultdict
from itertools import product
from pprint import pformat
from tqdm import tqdm
import random
import math
import os
import sys


def sample_one(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    else:
        print('Error: sample_one')
        exit()


def add_counts(word, topic, docid, amount, xcounts, ycounts):
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount
    if xcounts[topic] < 0 or xcounts[f'{word}|{topic}'] < 0:
        print('Error: xcounts in add_counts')
        exit()

    ycounts[docid] += amount
    ycounts[f'{word}|{docid}'] += amount
    if ycounts[docid] < 0 or ycounts[f'{word}|{docid}'] < 0:
        print('Error: ycounts in add_counts')
        exit()


def initialize(file):
    'xcorpus, ycorpus, xcounts, ycounts, num_words を file をもとに初期化'
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    all_word = set()
    for line in file:
        docid = len(xcorpus)
        topics = []
        words = line.strip().split()
        for word in words:
            all_word.add(word)
            topic = random.randint(0, NUM_TOPICS - 1)
            topics.append(topic)
            add_counts(word, topic, docid, 1, xcounts, ycounts)
        xcorpus.append(words)
        ycorpus.append(topics)
    return xcorpus, ycorpus, xcounts, ycounts, len(all_word)


if __name__ == '__main__':
    # path = '../../data/wiki-en-documents.word'.replace('/', os.sep)
    is_test = sys.argv[1:] == ["test"]
    path_test = '/Users/naoto_nakazawa/komachi_lab/nlptutorial/test/07-train.txt'.replace('/', os.sep)
    path_wiki = '/Users/naoto_nakazawa/komachi_lab/nlptutorial/data/wiki-en-documents.word'
    NUM_TOPICS = 2
    NUM_EPOCH = 5
    ALPHA = 0.01
    BETA = 0.01

    if is_test:
        file = open(path_test, 'r', encoding='utf8')
    else:
        file = open(path_wiki, 'r', encoding="utf8")
    xcorpus, ycorpus, xcounts, ycounts, num_words = initialize(file)

    for num_epoch in tqdm(range(NUM_EPOCH), 'Epoch'):
        ll = 0
        for i in tqdm(range(len(xcorpus)), 'xcorpus'):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(x, y, i, -1, xcounts, ycounts)
                probs = []
                for k in range(NUM_TOPICS):
                    xprob = (xcounts[f'{x}|{k}'] + ALPHA) / (xcounts[k] + ALPHA * num_words)
                    yprob = (ycounts[f'{k}|{i}'] + BETA) / (ycounts[i] + BETA * NUM_TOPICS)
                    probs.append(xprob * yprob)
                new_y = sample_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x, new_y, i, 1, xcounts, ycounts)
                ycorpus[i][j] = new_y
        print(ll)

    if is_test:
        path = "x_y_07-train.txt"
    else:
        path = "x_y.txt"
    with open(path, "w", encoding="utf-8") as fw:
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                if ycorpus[i][j] == 0:
                    print(f'{xcorpus[i][j]}:{ycorpus[i][j]}\t', end='', file=fw)
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                if ycorpus[i][j] == 1:
                    print(f'{xcorpus[i][j]}:{ycorpus[i][j]}\t', end='', file=fw)

    if is_test:
        path_x = 'xcounts_07-train.txt'
        path_y = 'ycounts_07-train.txt'
    else:
        path_x = 'xcounts.txt'
        path_y = 'ycounts.txt'
    with open(path_x, 'w', encoding='utf8') as f:
        f.write(pformat(sorted(xcounts.items(), key=lambda x: str(x[0])[-1])))
    with open(path_y, 'w', encoding='utf8') as f:
        f.write(pformat(sorted(ycounts.items(), key=lambda x: str(x[0])[-1])))
