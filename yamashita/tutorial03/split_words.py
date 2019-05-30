from collections import defaultdict
import sys
import math

model_path = sys.argv[1]
probabilities = defaultdict(lambda: 0)
input_path = sys.argv[2]

# モデル読み込み
with open(model_path, 'r', encoding='utf-8') as m_file:
    for line in m_file.readlines():
        words = line.rstrip().split('\t')
        if len(words) != 0:
            probabilities[words[0]] = float(words[1])


best_edge = []
best_score = []
lambda1 = 0.95
V = 1000000

with open(input_path, 'r', encoding='utf-8') as i_file:
    for line in i_file:
        # 前向きステップ
        line = line.rstrip()
        best_edge = defaultdict()
        best_score = defaultdict()
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            best_score[word_end] = 10000000000
            for word_begin in range(word_end):
                word = line[word_begin:word_end]
                if word not in probabilities and len(word) != 1:
                    continue
                prob = lambda1 * probabilities[word] + (1-lambda1)/V
                my_score = best_score[word_begin] + -math.log(prob)
                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = [word_begin, word_end]

        # 後向きステップ
        words = []
        next_edge = best_edge[len(best_edge)-1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(' '.join(words))
