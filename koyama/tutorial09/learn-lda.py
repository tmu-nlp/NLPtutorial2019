from collections import defaultdict
import random
import math
from tqdm import tqdm

NUM_TOPICS = 2
epoch = 10
#data_path = '../../test/07-train.txt'
data_path = '../../data/wiki-en-documents.word'

def ADDCOUNTS(word, topic, docid, amount):
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount
    ycounts[docid] += amount
    ycounts[f'{topic}|{docid}'] += amount

def SAMPLEONE(probs):
    z = sum(probs)
    remaining = random.random()*z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    raise Exception('Error at SAMPLEONE')

def prob_of_topic_k(x, k, Y):
    alpha = 0.02
    beata = 0.02
    Nx = len(xcorpus)
    Ny = len(ycorpus)
    P_x_given_k = (xcounts[f'{x}|{k}'] + alpha)/(xcounts[k] + alpha*Nx)
    P_y_given_Y = (ycounts[f'{k}|{Y}'] + beata)/(ycounts[Y] + beata*Ny)
    return P_x_given_k*P_y_given_Y

xcorpus = []
ycorpus = []
xcounts = defaultdict(lambda: 0)
ycounts = defaultdict(lambda: 0)
with open(data_path, 'r') as data_file:
    for line in data_file:
        docid = len(xcorpus)
        words = line.rstrip().split(' ')
        topics = []
        for word in words:
            topic = random.randint(0, NUM_TOPICS - 1)
            topics.append(topic)
            ADDCOUNTS(word, topic, docid, 1)
        xcorpus.append(words)
        ycorpus.append(topics)

for _ in tqdm(range(epoch)):
    ll = 0
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            x = xcorpus[i][j]
            y = ycorpus[i][j]
            ADDCOUNTS(x, y, i, -1)
            probs = []
            for k in range(NUM_TOPICS):
                probs.append(prob_of_topic_k(x, k, i))
            new_y = SAMPLEONE(probs)
            ll += math.log(probs[new_y])
            ADDCOUNTS(x, new_y, i, 1)
            ycorpus[i][j] = new_y

with open('answer-file', 'w') as ans_file:
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            x = xcorpus[i][j]
            y = ycorpus[i][j]
            print(f'{x}_{y}', end=' ', file=ans_file)
        print(file=ans_file)
