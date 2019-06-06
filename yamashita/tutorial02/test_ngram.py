import sys
import math
from collections import defaultdict

model_path = sys.argv[1]
test_path = sys.argv[2]
N = int(sys.argv[3])
mode = sys.argv[4]
# linear_lambda_1 = 0.95
# linear_lambda_2 = 0.80
linear_lambda_step = 5
V = 1000000
# W = 0
# H = 0
probs = defaultdict(lambda: 0)

with open(model_path, 'r', encoding='utf-8') as m_file:
    for line in m_file.readlines():
        line = line.strip('\n').split('\t')
        probs[line[0]] = float(line[1])

with open(test_path, 'r', encoding='utf-8') as t_file:
    lines = t_file.readlines()
    if mode == 'linear' and N != 2:
        print('線形補間はbigramのみしかサポートしていません')
    elif mode == 'linear' and N == 2:
        for lambda100_1 in range(0, 100, linear_lambda_step):
            for lambda100_2 in range(0, 100, linear_lambda_step):
                linear_lambda_1 = float(lambda100_1)/100
                linear_lambda_2 = float(lambda100_2)/100
                W = 0
                H = 0
                for line in lines:
                    words = line.strip('\n').split(' ')
                    words.insert(0, '<s>')
                    words.append('</s>')
                    W += len(words)
                    for i in range(1, len(words)):
                        P1 = linear_lambda_1 * \
                            probs[words[i]] + (1-linear_lambda_1)/V
                        P2 = linear_lambda_2 * \
                            probs[' '.join(words[i-1:i+1])] + \
                            (1-linear_lambda_2)*P1
                        H += -math.log2(P2)
                        # print(P1, P2)
                print(
                    f'lambda1 = {linear_lambda_1}\tlambda2 = {linear_lambda_2}\tentropy = {H/W:.6f}')
                # print(H, W)
    # elif mode == 'wb':
    #     for line in t_file.readlines():
    #         words = line.strip().split(' ')
    #         words.insert(0, '<s>')
    #         words.append('</s>')
    #         # for i in range(N-1, len(words)):

    #         #     for j in range(1, N):

    else:
        print('存在しないmodeです')
