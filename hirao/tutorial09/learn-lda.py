import numpy as np
from collections import defaultdict

NUM_TOPICS = 2
ITER = 1

class LDA:
    def __init__(self):
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(int)
        self.ycounts = defaultdict(int)
        self.init()

    def init(self):
        train_path = "../../data/wiki-en-documents.word"
        for line in open(train_path):
            doc_id = len(self.xcorpus)
            topics = []
            words = line.split()
            for word in words:
                topic = np.random.randint(0, NUM_TOPICS)
                topics.append(topic)
                self.add_counts(word, topic, doc_id, 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)

    def add_counts(self, word, topic, doc_id, amount):
        self.xcounts[f"{topic}"] += amount
        self.xcounts[f"{word}|{topic}"] += amount
        self.ycounts[f"{doc_id}"] += amount
        self.ycounts[f"{topic}|{doc_id}"] += amount

    def sample_one(self, probs):
        z = sum(probs)
        remaining = np.random.rand() * z
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i

    def sampling(self):
        for _ in range(ITER):
            ll = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, -1)
                    probs = []
                    for k in range(NUM_TOPICS):
                        p_xk = self.xcounts[f"{x}|{k}"] / (self.xcounts["k"] + 1)
                        p_ky = self.ycounts[f"{k}|{y}"] / (self.ycounts["y"] + 1)
                        probs.append(p_xk * p_ky)
                    new_y = self.sample_one(probs)
                    ll += np.log(probs[new_y])
                    self.add_counts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y
            print(ll)
        for xs, ys in zip(self.xcorpus, self.ycorpus):
            for x, y in zip(xs, ys):
                print(x, y)

if __name__ == "__main__":
    lda = LDA()
    lda.sampling()
