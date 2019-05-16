from collections import defaultdict
import numpy as np
import sys
from math import log2

def word_segmentation(filename: str) -> None:
    unk = 0.05 # 未知語の確率
    N = 1e6    # 未知語を含む語彙数

    # 1-gram モデルの読み込み
    probabilities = defaultdict(int)
    for line in open("./model_file.txt", "r", encoding="utf-8"):
        line = line.strip().split("\t")
        probabilities[line[0]] = float(line[1])

    for line in open(filename, "r", encoding="utf-8"):
        # 前向きステップ
        line = line.strip("\n")
        length = len(line) + 1
        best_edge = [None for _ in range(length)]
        best_score = [0 for _ in range(length)]

        for word_end in range(1, length):
            best_score[word_end] = 1e10

            for word_begin in range(word_end):
                word = line[word_begin:word_end]

                if word in probabilities or len(word) == 1:
                    prob = (1 - unk) * probabilities[word] + unk / N
                    my_score = best_score[word_begin] + -log2(prob)

                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
            
        # 後向きステップ
        words = []
        next_edge = best_edge[-1] # type: Tuple[int]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(" ".join(words))

if __name__ == "__main__":
    args = sys.argv
    word_segmentation(args[1])

"""
$ ./gradews.pl data/wiki-ja-test.word result                                                                                                       [~/works/nlp_tutorial/tutorial03]
Sent Accuracy: 23.81% (20/84)
Word Prec: 71.88% (1943/2703)
Word Rec: 84.22% (1943/2307)
F-meas: 77.56%
Bound Accuracy: 86.30% (2784/3226)
"""