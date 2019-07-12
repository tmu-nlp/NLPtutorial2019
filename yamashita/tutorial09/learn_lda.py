from tqdm import tqdm
import random
import math
from collections import defaultdict


def initialize(train_file, num_topics):
    with open(train_file, 'r', encoding='utf-8') as tr_file:
        xcorpus = []
        ycorpus = []
        xcounts = defaultdict(int)
        ycounts = defaultdict(int)
        num_wordtype = set()
        for line in tr_file:
            doc_id = len(xcorpus)
            words = line.strip().split()
            topics = []
            for word in words:
                topic = random.randint(0, num_topics)
                topics.append(topic)
                xcounts, ycounts = add_counts(
                    xcounts, ycounts, word, topic, doc_id, 1)
                num_wordtype.add(word)
            xcorpus.append(words)
            ycorpus.append(topics)
    return xcorpus, ycorpus, xcounts, ycounts, len(num_wordtype)


def add_counts(xcounts, ycounts, word, topic, doc_id, amount):
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount
    ycounts[doc_id] += amount
    ycounts[f'{topic}|{doc_id}'] += amount
    return xcounts, ycounts


def sampleone(probs):
    z = sum(probs)
    remaining = random.random()*z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == (len(probs)-1):
            print('error')
            exit()


def main():
    test_file = '../../data/wiki-en-documents.word'
    # test_file = '../../test/07-train.txt'
    epoch = 1
    alpha = 0.01
    beta = 0.01
    num_topics = 2
    xcorpus, ycorpus, xcounts, ycounts, num_wordtype = initialize(
        test_file, num_topics)
    for _ in tqdm(range(epoch)):
        ll = 0
        for i in tqdm(range(len(xcorpus))):
            for j in tqdm(range(len(xcorpus[i]))):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                xcounts, ycounts = add_counts(xcounts, ycounts, x, y, i, -1)
                probs = []
                for k in tqdm(range(num_topics)):
                    p_xk = (xcounts[f'{x}|{k}'] + alpha) / \
                        (xcounts[k] + alpha * num_wordtype)
                    p_ky = (ycounts[f'{k}|{i}'] + beta) / \
                        (ycounts[i] + beta * num_topics)
                    probs.append(p_xk * p_ky)

                new_y = sampleone(probs)
                ll += math.log(probs[new_y])
                add_counts(xcounts, ycounts, x, new_y, i, 1)
                ycorpus[i][j] = new_y
        print(ll)

        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                print(f'{xcorpus[i][j]}_{ycorpus[i][j]}', end=' ')
            print()


if __name__ == '__main__':
    main()
