import sys
from collections import defaultdict
import math

probabilites = defaultdict(lambda: 0)

with open('model-file.txt', 'r') as model_file:
    for line in model_file:
        line = line.strip(). split()
        probabilites[line[0]] = float(line[1])

lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V = 1000000
W = 0
H = 0
unk = 0

with open('../data/wiki-en-test.word', 'r') as test_file:
    for line in test_file:
        words = line.strip().split()
        words.append("</s>")
        for w in words:
            W += 1
            P = lambda_unk / V
            if w in probabilites:
                P += lambda_1 * probabilites[w]
            else:
                unk += 1
            H += -math.log(P, 2)

print("entropy  = " + str(H/W))
print("coverage = " + str((W - unk) / W))
