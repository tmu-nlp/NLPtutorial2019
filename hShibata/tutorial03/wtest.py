import os
from collections import defaultdict
import sys
import math
import json
import shutil
import nGram
import subprocess

def ViterbiDivision(path):
    with open(path, "r") as f:
        with open("out.word", "w") as fo:
            for line in f:
                line = line.strip()
                Graph = {}
                # create a graph
                for i in range(0, len(line)):
                    tG = {}
                    for j in range(i+1, len(line)+1):
                        qw = line[i:j]
                        ln = -math.log(nGram.probWord([qw]))
                        if ln > math.log(nGram.nVirtualWords):
                            ln = ln*len(qw) # this is an unknown word. multiply length to restrict the length of unknown words to 1 on Viterbi algorithm.

                        tG[j] = ln

                    Graph[i] = tG
                # Excute Viterbi algorithm.
                best_score = {}
                best_edge = {}
                best_score[0] = 0
                for i in range(1, len(line)+1):
                    best_score[i] = math.inf
                    for j in range(0, i):
                        score = best_score[j] + Graph[j][i]
                        if score < best_score[i]:
                            best_score[i] = score
                            best_edge[i] = j
                words = []
                i = len(line)
                while True:
                    j = best_edge[i]
                    words.append(line[j:i])
                    i = j
                    if i == 0:
                        break
                words.reverse()


                oStr = ""
                for w in words:
                    oStr = oStr + w + " "
                print(oStr, file=fo)


a = 3
if a == 1:
    with open("../../test/04-model.txt", "r") as f:
        nGram.gN = 1
        nGram.mnGram = nGram.c_nGram(0.98)
        nGram.mnGram.prob = 1/nGram.nVirtualWords
    
        for line in f:
            line = line.strip()
            line = line.split("\t")
            print(line)
            nGram.mnGram.w_prev[line[0]].prob = float(line[1])
    
    ViterbiDivision("../../test/04-input.txt")

elif a == 2:
    nGram.gN = 1
    nGram.mnGram = nGram.c_nGram(0.95)
    nGram.Train("04-train.word")
    ViterbiDivision("04-input.txt")


# test 2, practice of tutorial 3 using a wikipedia dataset.
elif a == 3:
    nGram.gN = 1
    nGram.nVirtualWords = 1000000
    nGram.mnGram = nGram.c_nGram(0.95)
    nGram.Train("../../data/wiki-ja-train.word")
    ViterbiDivision("../../data/wiki-ja-test.txt")
    subprocess.call(["perl", "../gradews.pl","../../data/wiki-ja-test.word","out.word"])

    # Sent Accuracy: 0.00% (/84)
    # Word Prec: 79.78% (1925/2413)
    # Word Rec: 83.44% (1925/2307)
    # F-meas: 81.57%
    # Bound Accuracy: 88.72% (2862/3226)
else:
    with open("big-ws-model.txt", "r") as f:
        nGram.gN = 1
        nGram.mnGram = nGram.c_nGram(0.98)
        nGram.mnGram.prob = 1/nGram.nVirtualWords
    
        for line in f:
            line = line.strip()
            line = line.split("\t")
            nGram.mnGram.w_prev[line[0]].prob = float(line[1])

    ViterbiDivision("../../data/wiki-ja-test.txt")
    subprocess.call(["perl", "../gradews.pl","../../data/wiki-ja-test.word","out.word"])
    # Sent Accuracy: 0.00% (/84)
    # Word Prec: 74.24% (1919/2585)
    # Word Rec: 83.18% (1919/2307)
    # F-meas: 78.45%
    # Bound Accuracy: 86.05% (2776/3226)
