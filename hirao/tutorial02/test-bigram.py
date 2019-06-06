# test-bigram.py
import math
from collections import defaultdict

DEBUG = False
# データパス
model_path = "train-input-model.txt" if DEBUG else "wiki-en-train-model.txt"
test_data_path =  "../../test/02-test-input.txt" if DEBUG else "../../data/wiki-en-test.word"

lambda_1 = 0.95
lambda_2 = 0.80
V = 1000000
w = 0
h = 0

probs = defaultdict(lambda: 0)
with open(model_path) as f:
    for line in f:
        ngram, probability = line.strip().split("\t")
        probs[ngram] = float(probability)
probs

for lambda_1 in range(0, 100, 5):
    lambda_1 /= 100
    for lambda_2 in range(0, 100, 5):
        lambda_2 /= 100
        with open(test_data_path) as f:
            for line in f:
                words = line.strip().split()
                # 終端記号を追加
                words.append("</s>")
                # 文頭記号を追加
                words.insert(0, "<s>")
                w += len(words)
                for i in range(len(words) - 1):
                    p1 = lambda_1 * probs[words[i+1]] + (1 - lambda_1) / V
                    p2 = lambda_2 * probs["{} {}".format(words[i], words[i+1])] + (1 - lambda_2) * p1
                    h += -math.log(p2, 2)
            print("lambda_1: {}, lambda_2: {}, entropy: {}".format(lambda_1, lambda_2, h/w))