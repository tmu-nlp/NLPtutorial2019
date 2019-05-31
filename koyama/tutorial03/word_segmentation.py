from collections import defaultdict
import math

probabilities = defaultdict(lambda: 0)

with open('model-file.txt', 'r') as model_file:
    for line in model_file:
        line = line.strip().split()
        probabilities[line[0]] = float(line[1])

lambda_1 = 0.95
V = 1000000

with open('nlptutorial/data/wiki-ja-test.txt', 'r') as input_file,\
     open('my_answer.word', 'w') as ans_file:
    for line in input_file:
        best_edge = defaultdict(lambda: 0)
        best_score = defaultdict(lambda: 0)
        line = line.strip()
        best_edge[0] = "NULL"
        best_score[0] = 0
        for word_end in range(1, len(line) + 1):
            best_score[word_end] = 10000000000
            for word_begin in range(word_end):
                word = line[word_begin:word_end]
                if word in probabilities or len(word) == 1:
                    prob = lambda_1*probabilities[word] + (1 - lambda_1)/V
                    my_score = best_score[word_begin] + -math.log(prob, 2)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = [word_begin, word_end]
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge != "NULL":
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(' '.join(words), file=ans_file)
