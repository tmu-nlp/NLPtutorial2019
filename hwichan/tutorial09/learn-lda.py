from collections import defaultdict
import numpy as np
import os.path
import random
import math
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords
import snowballstemmer
stopWords = stopwords.words('english')
stemmer = snowballstemmer.stemmer('english')


class initialize_topic_model:
    def __init__(self, filename):
        self.filename = filename
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)
        self.topics_vector = []
        self.TOPICS = 7
        self.different_word = []

    def initialize(self):
        first_time = 1
        adder = add_count(self.xcounts, self.ycounts)
        for docid, line in enumerate(open(self.filename)):
            rline = line.rstrip("\n")
            # words = rline.split(" ")
            words = []
            for word in rline.split(" "):
                if self.is_stopword(word) or len(word) == 1:
                    continue
                words.append(stemmer.stemWord(word))
            self.different_word += words

            topics_vector = []
            for word in words:
                topic = random.randint(0, self.TOPICS - 1)  # 単語のトピックをランダムに初期化
                topics_vector.append(topic)
                adder.add_counter(word, topic, docid, 1)

            self.xcorpus.append(words)
            self.ycorpus.append(topics_vector)

        self.different_word = set(self.different_word)

    def is_stopword(self, word):
        if word in stopWords:
            return True
        else:
            return False


class add_count:
    def __init__(self, xcounts, ycounts):
        self.xcounts = xcounts
        self.ycounts = ycounts


    def add_counter(self, word, topic, docid, amount):
        self.xcounts[topic] += amount
        self.xcounts[(word, topic)] += amount

        self.ycounts[docid] += amount
        self.ycounts[(topic, docid)] += amount


class Sampling:
    def __init__(self, xcorpus, ycorpus, xcounts, ycounts):
        self.iteration = 1000
        self.xcorpus = xcorpus
        self.ycorpus = ycorpus
        self.alpha = 0.01
        self.beta = 0.03
        self.xcounts = xcounts
        self.ycounts = ycounts
        self.adder = add_count(self.xcounts, self.ycounts)

    def sampling(self, TOPICS, different_word):
        for i in tqdm(range(0, self.iteration), desc='iteration'):
            self.sampler(TOPICS, different_word)

    def sampler(self, TOPICS, different_word):
        ll = 0
        for i in tqdm(range(len(self.xcorpus)), desc='xcorpus'):
            for j in range(len(self.xcorpus[i])):
                x = self.xcorpus[i][j]  # 単語
                y = self.ycorpus[i][j]  # トピック
                self.adder.add_counter(x, y, i, -1)
                probs = []
                for k in range(TOPICS):
                    p_x_y = (1.0 * self.xcounts[(x, k)] + self.alpha) / (self.xcounts[k] + self.alpha * len(different_word))
                    p_y_Y = (1.0 * self.ycounts[(y, i)] + self.beta) / (self.ycounts[i] + self.beta * TOPICS)
                    probs.append(p_x_y * p_y_Y)
                new_y = self.sampleOne(probs)
                ll += math.log(probs[new_y])
                self.adder.add_counter(x, new_y, i ,1)
                self.ycorpus[i][j] = new_y
        print(ll)

    def sampleOne(self, probs):
        z = sum(probs)
        remaining = random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i


if __name__ == "__main__":
    # filename = "../../test/07-train.txt"
    filename = "../../data/wiki-en-documents.word"
    model = initialize_topic_model(filename)
    model.initialize()
    sample = Sampling(model.xcorpus, model.ycorpus, model.xcounts, model.ycounts)
    sample.sampling(model.TOPICS, model.different_word)

    dict0 = defaultdict(lambda: 0)
    dict1 = defaultdict(lambda: 0)
    dict2 = defaultdict(lambda: 0)
    dict3 = defaultdict(lambda: 0)
    dict4 = defaultdict(lambda: 0)
    dict5 = defaultdict(lambda: 0)
    dict6 = defaultdict(lambda: 0)
    for i in range(len(sample.xcorpus)):
        for j in range(len(sample.xcorpus[i])):
            word = sample.xcorpus[i][j]
            topic = sample.ycorpus[i][j]
            if topic == 0:
                dict0[word] += 1
            elif topic == 1:
                dict1[word] += 1
            elif topic == 2:
                dict2[word] += 1
            elif topic == 3:
                dict3[word] += 1
            elif topic == 4:
                dict4[word] += 1
            elif topic == 5:
                dict5[word] += 1
            elif topic == 6:
                dict6[word] += 1

    with open('dict0.txt', 'w') as out0, open('dict1.txt', 'w') as out1,\
       open('dict2.txt', 'w') as out2, open('dict3.txt', 'w') as out3,\
       open('dict4.txt', 'w') as out4, open('dict5.txt', 'w') as out5,\
       open('dict6.txt', 'w') as out6:

       for k, v in sorted(dict0.items(), key=lambda x: -x[1]):
           print(k, v, file=out0)
       for k, v in sorted(dict1.items(), key=lambda x: -x[1]):
           print(k, v, file=out1)
       for k, v in sorted(dict2.items(), key=lambda x: -x[1]):
           print(k, v, file=out2)
       for k, v in sorted(dict3.items(), key=lambda x: -x[1]):
           print(k, v, file=out3)
       for k, v in sorted(dict4.items(), key=lambda x: -x[1]):
           print(k, v, file=out4)
       for k, v in sorted(dict5.items(), key=lambda x: -x[1]):
           print(k, v, file=out5)
       for k, v in sorted(dict6.items(), key=lambda x: -x[1]):
           print(k, v, file=out6)
