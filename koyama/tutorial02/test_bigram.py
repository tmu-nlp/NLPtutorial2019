from collections import defaultdict
import math

probs = defaultdict(lambda: 0)

with open('model-file.txt', 'r') as model_file:
    for line in model_file:
        line = line.strip().split()
        probs[line[0]] = float(line[1])

V = 1000000
min_entropy = float('inf')

with open('../../data/wiki-en-test.word', 'r') as test_file:
    text = test_file.readlines()
    for lambda_1 in range(1, 100, 5):
        for lambda_2 in range(1, 100, 5):
            lambda_1 = lambda_1 / 100
            lambda_2 = lambda_2 / 100
            W = 0
            H = 0
            for line in text:
                words = line.strip().split()
                words.append('</s>')
                #words.insert(0, '<s>')
                W += len(words) - 1
                for i in range(1, len(words)):
                    P1 = lambda_1 * probs[words[i]] + (1 - lambda_1) / V
                    P2 = lambda_2 * probs[' '.join(words[i - 1: i + 1])] + (1 - lambda_2) * P1
                    H += -math.log(P2, 2)
            entropy = H / W
            if entropy < min_entropy:
                min_entropy = entropy
                min_lambda1 = lambda_1
                min_lambda2 = lambda_2

print('最小のエントロピー:' + str(min_entropy))
print('lambda1:' + str(min_lambda1))
print('lambda2:' + str(min_lambda2))
