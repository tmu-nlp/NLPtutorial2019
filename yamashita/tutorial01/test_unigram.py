import sys
from collections import defaultdict
import math

model_file_path = sys.argv[1]
test_file_path = sys.argv[2]
probabilities = defaultdict(lambda:0)


# モデル読み込み
with open(model_file_path,'r',encoding='utf-8') as model_file:
    for line in model_file.readlines():
        words = line.strip().split(' ')
        if len(words) != 0:
            probabilities[words[0]] = float(words[1])



# 評価・結果表示
lambda_1 = 0.95
lambda_unknown = 1 - lambda_1
V = 1000000
W = 0
H = 0
unk = 0

with open(test_file_path,'r',encoding='utf-8') as test_file:
    for line in test_file.readlines():
        words = line.strip().split(' ')
        words.append('</s>')
        for word in words:
            W += 1
            P = lambda_unknown / V
            if word in probabilities:
                P += lambda_1 * probabilities[word]
            else:
                unk += 1
            H += - math.log2(P)

print('entropy = {}'.format(float(H)/W))
print('coverage = {}'.format(float(W - unk)/W))