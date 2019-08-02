import os
import sys
import math
import random
from tqdm import tqdm
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
random.seed(42)


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stdout.write("\33[92m" + text + "\33[0m")


def add_counts(xcounts, ycounts, word, topic, docid, amount):
    """ #09 p23 """
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount
    ycounts[docid] += amount
    ycounts[f'{topic}|{docid}'] += amount


def initialize(train_file, num_topics):
    """ #09 p22 """
    xcorpus, ycorpus = [], []
    xcounts, ycounts = defaultdict(int), defaultdict(int)
    num_wordtype = set()
    for line in open(train_file):
        docid = len(xcorpus)
        words = line.split()
        topics = []
        for word in words:
            topic = random.randint(0, num_topics)
            topics.append(topic)
            add_counts(xcounts, ycounts, word, topic, docid, 1)
            num_wordtype.add(word)
        xcorpus.append(words)
        ycorpus.append(topics)
    return xcorpus, ycorpus, xcounts, ycounts, len(num_wordtype)


def sampleone(probs):
    """ #09 p14 """
    z = sum(probs)
    remaining = random.random() * z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    message('error')
    exit()


def sample(test_path, epoch=1, α=0.01, β=0.01, num_topics=2):
    """ #09 p23 """
    xcorpus, ycorpus, xcounts, ycounts, num_wordtype = initialize(
        test_path, num_topics)
    for _ in tqdm(range(epoch)):
        ll = 0
        for i in tqdm(range(len(xcorpus))):
            for j in tqdm(range(len(xcorpus[i]))):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(xcounts, ycounts, x, y, i, -1)
                probs = []
                for k in tqdm(range(num_topics)):
                    p_xk = (xcounts[f'{x}|{k}'] + α) / \
                        (xcounts[k] + α * num_wordtype)
                    p_ky = (ycounts[f'{k}|{i}'] + β) / \
                        (ycounts[i] + β * num_topics)
                    probs.append(p_xk * p_ky)
                new_y = sampleone(probs)
                ll += math.log(probs[new_y])
                add_counts(xcounts, ycounts, x, new_y, i, 1)
                ycorpus[i][j] = new_y
        message(f'll={ll}\n', CR=True)
    message('\n')
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            print(f'{xcorpus[i][j]}_{ycorpus[i][j]}', end=' ')
        print()


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        message('test')
        sample(test_path='../../test/07-train.txt')
    else:
        message('main')
        sample(test_path='../../data/wiki-en-documents.word')
