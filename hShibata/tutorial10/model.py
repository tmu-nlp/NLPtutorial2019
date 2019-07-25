import os
from collections import defaultdict
import sys
import math
import json
import shutil
import subprocess
import random
import operator
import math
import numpy as np

prob_gram = {}
symbol = set()
t_dict = {}
aa = [1, 2, 1, 1]
t_dict[tuple([2, 2])] = 1
t_dict[tuple([2, 3])] = 31
t_dict[tuple([1, 2])] = 2
for a, b in t_dict.items():
    i, j = a
    print(i, j, b)
quit()
#pathIn = "input2"
#pathGrammar = "grammar"
pathIn = "../../test/08-input.txt"
pathGrammar = "../../test/08-grammar.txt"
pathOut = "out.txt"
with open(pathGrammar, "r") as f:
    for rule in f:
        lhs, rhs, prob = rule.strip().split("\t")
        symbol.add(lhs)
        symbols = rhs.split(" ")
        if len(symbols) == 1:
            prob_gram[lhs + "->" + rhs] = float(prob)
        else:
            prob_gram[lhs + "->(" + symbols[0] + ", " +
                      symbols[1] + ")"] = float(prob)
print(symbol)
print(prob_gram)
with open(pathIn, "r") as f:
    with open(pathOut, "w") as fo:
        for line in f:
            words = line.strip().split(" ")
            Score_best = []
            n = len(words)
            edge_best = []
            for i in range(0, n):
                ti = []
                ei = []
                for j in range(0, n-i):
                    ti.append({})
                    ei.append({})
                Score_best.append(ti)
                edge_best.append(ei)

            for j in range(0, n):
                min_lnProb = float("inf")
                k = 0
                for l in symbol:
                    gram = l + "->" + words[j]
                    if gram in prob_gram:
                        Score_best[0][j][tuple(k, l)] = -math.log(prob_gram[gram])
                        Score_best[0][j][tuple(k, l)] = words[j]
                        k = k + 1

            # forward path
            for i in range(1, n):
                for j in range(0, n-i):
                    for k in range(0, i):
                        for l in symbol:
                            ip = k
                            jp = j
                            ipp = i - k - 1
                            jpp = j + k + 1
                            # print("c:",i,j,k,l,mp,mpp)
                            min_lnProb = float("inf")
                            edge = []
                            for key1, val1 in Score_best[ip][jp].items():
                                kp, lp = key1
                                for key2, val2 in Score_best[ipp][jpp].items():
                                    kpp, lpp = key2
                                    gram = l + "->(" + lp + ", " + lpp + ")"
                                    if gram in prob_gram:
                                        lnProb = - math.log(
                                                prob_gram[gram]) + val1 + val2
                                        if lnProb < min_lnProb:
                                            min_lnProb = lnProb
                                            edge = [key1, key2]
                            # print(l,min_lnProb)
                            if min_lnProb != float("inf"):
                                Score_best[i][j][tuple(k, l)] = min_lnProb
                                edge_best[i][j][tuple(k, l)] = edge

            # backward path
            str_t = ""

            def rec(i, j, k, l):
                if i != 0:
                    print("rec1:", i, j, k, l)
                    kp, kpp, lp, lpp = edge_best[i][j][k][l]
                    ip = k
                    jp = j
                    ipp = i - k - 1
                    jpp = j + k + 1
                    str_a = ""
                    str_a = str_a + " (" + lp + " "
                    kk = rec(ip, jp, kp, lp)
                    str_a = str_a + kk
                    print("rec2:", ip, jp, kp, lp, kk)
                    str_a = str_a + ")"

                    str_a = str_a + " (" + lpp + " "
                    kk = rec(ipp, jpp, kpp, lpp)
                    print("rec3:", ipp, jpp, kpp, lpp, kk)
                    str_a = str_a + kk
                    str_a = str_a + ")"
                    return str_a
                else:
                    print("last:", edge_best[i][j][0][l])
                    return edge_best[i][j][0][l]

            min_lnProb = float("inf")
            k = 0
            l = ""
            i = n-1
            for kt in range(0, n-1):
                for lt in symbol:
                    if min_lnProb > Score_best[i][0][kt][lt]:
                        min_lnProb = Score_best[i][0][kt][lt]
                        k = kt
                        l = lt

            print(i, j, k, l)
            print(edge_best)
            print(Score_best[i][0])
            str_t = str_t + "(" + l + " "
            str_t = str_t + rec(n-1, 0, k, l)
            str_t = str_t + ")"
            print(str_t, file=fo)

            # for it in range(0, n-1):
            # print(Score_best[it])
            # print(edge_best[it])
