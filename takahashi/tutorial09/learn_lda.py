from collections import defaultdict as dd
from random import randint, uniform
from math import log
from sys import exit
from tqdm import tqdm

NUM_TOPICS = 3
FILEPATH = "../files/data/wiki-en-documents.word"

ALPHA = 0.01
BETA = 0.01

def sample_one(probs):
    z = sum(probs)
    remaining = uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    exit()

def add_counts(word, topic, doc_id, amounts, xcnts, ycnts):
    xcnts[f"{topic}"] += 1
    xcnts[f"{word}|{topic}"] += 1

    ycnts[f"{doc_id}"] += 1
    ycnts[f"{topic}|{doc_id}"] += 1


def initialize():
    xcorpus, ycorpus = [], []
    xcounts, ycounts = dd(int), dd(int)
    unique_words = set()

    for line in open(FILEPATH, "r", encoding="utf-8"):
        doc_id = len(xcorpus)
        words = line.strip().split()
        topics = []

        for word in words:
            unique_words.add(word)
            topic = randint(0, NUM_TOPICS-1)
            topics.append(topic)

            add_counts(word, topic, doc_id, 1, xcounts, ycounts)

        xcorpus.append(words)
        ycorpus.append(topics)

    num_words = len(unique_words)
    return xcorpus, ycorpus, xcounts, ycounts, num_words

def sampling():
    xcorps, ycorps, xcnts, ycnts, num_words = initialize()
    ll = 0

    for i in tqdm(range(len(xcorps)), desc="sent"):
        for j in range(len(xcorps[i])):
            x, y = xcorps[i][j], ycorps[i][j]
            add_counts(x, y, i, -1, xcnts, ycnts)

            probs = []
            for k in range(NUM_TOPICS):
                x_prob = (xcnts[f"{x}|{k}"] + ALPHA) / (xcnts[k] + ALPHA * num_words)
                y_prob = (ycnts[f"{k}|{i}"] + BETA) / (ycnts[i] + BETA * NUM_TOPICS)
                probs.append(x_prob * y_prob)
        new_y = sample_one(probs)
        ll += log(probs[new_y])
        add_counts(x, new_y, i, 1, xcnts, ycnts)
        ycorps[i][j] = new_y

    print(ll)

if __name__ == "__main__":
    sampling()