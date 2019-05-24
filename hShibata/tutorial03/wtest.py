import os
from collections import defaultdict
import sys
import math
import json
import shutil


class nGram:
    def __init__(self, prob_t, r):
        self.prob = prob_t
        self.count = 0
        self.weight = 0
        self.w_prev = defaultdict(lambda: nGram(prob_t*r,r))

mDictionary = {}

#create graph edges.
with open("../../test/04-model.txt", "r") as f:
    for line in f:
        line = line.strip()
        line = line.split("\t")
        print(line)
        prob = float(line[1])
        mDictionary[line[0]] = -math.log2(prob)
            
with open("../../test/04-input.txt", "r") as f:
    for line in f:
        line = line.strip()
        Graph = {}
        #create a graph
        for i in range(0,len(line)):
            tG = {}
            for j in range(1,len(line)+1):
                qw = line[i:j]
                if qw in mDictionary:
                    tG[j] = mDictionary[qw]
                else:
                    tG[j] = math.inf
            Graph[i] = tG

        #Excute Bitabi algorithm.
        best_score = {}
        best_edge = {}
        best_score[0] = 0
        for i in range(1,len(line)+1):
            best_score[i] = math.inf
            for j in range(0,i):
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
        words = words.reverse()

        print(words)
