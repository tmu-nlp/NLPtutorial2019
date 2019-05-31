import subprocess
import shutil
import json
import math
import os
from collections import defaultdict
import sys
import nGram


class c_PE:
    def __init__(self, weight):
        self.count = 0
        self.prob = 0
        self.weight = weight
        self.pRand = defaultdict(lambda: c_PE(weight))


gPE_given_pos = c_PE(0.95)
gN = 10**6
gPosSet = set()


def train(path):
    strPos = ""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            word_poses = ["<s>_<s>"] + line.split(" ") + ["</s>_</s>"]
            for word_pos in word_poses:
                word, pos = word_pos.split("_")
                strPos += pos + " "  # create a dataset for pos bi-gram.
                gPE_given_pos.count += 1
                gPE_given_pos.pRand[pos].count += 1
                gPE_given_pos.pRand[pos].pRand[word].count += 1
                gPosSet.add(pos)

    # build a bi-gram model for the POS Markov chain.
    with open("tempPOS.word", "w") as f:
        print(strPos[:-1], file=f)

    nGram.mnGram = nGram.c_nGram(1)  # no smoothing
    nGram.gN = 2  # bi-gram.
    nGram.Train("tempPOS.word")

    # build a uni-gram model for the visible variable probability.
    for pos, gvn_pos in gPE_given_pos.pRand.items():
        for word, w_gvn_pos in gvn_pos.pRand.items():
            w_gvn_pos.prob = w_gvn_pos.count/gvn_pos.count


def probW(word, pos):
    if pos not in gPE_given_pos.pRand:
        raise Exception("unknown pos was found. this model is not supported.")

    gvn_pos = gPE_given_pos.pRand[pos]
    weight = gPE_given_pos.weight
    if word in gvn_pos.pRand:
        w_gvn_pos = gvn_pos.pRand[word]
        return weight*w_gvn_pos.prob + (1-weight)*1/gN
    else:
        return (1-weight)*1/gN


def ViterbiDivision(path):
    with open(path, "r") as f:
        oStr = ""
        for line in f:
            line = line.strip()
            words = ["<s>"] + line.split(" ") + ["</s>"]
            Graph = {}
            # create a graph
            for i in range(1, len(words)):
                if i == 1:
                    posSet1 = set({"<s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"})
                elif i == len(words)-1:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = set({"</s>"})
                else:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"})

                word = words[i]

                for pos1 in posSet1:
                    tG = {}
                    for pos2 in posSet2:
                        ln = - math.log(nGram.probWord([pos1,pos2])+0.1**100) - math.log(probW(word, pos2))

                        tG[str(i) + ":" +  pos2] = ln

                    Graph[str(i-1) + ":" +  pos1] = tG

            # Excute Viterbi algorithm.
            best_score = {}
            best_edge = {}
            best_score[str(0) + ":" + "<s>"] = 0
            for i in range(1, len(words)):
                if i == 1:
                    posSet1 = set({"<s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"})
                elif i == len(words)-1:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = set({"</s>"})
                else:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"})

                for pos2 in posSet2:
                    i2 = str(i) + ":" + pos2
                    best_score[i2] = math.inf
                    for pos1 in posSet1:
                        i1 = str(i-1)  + ":" +  pos1
                        score = best_score[i1] + Graph[i1][i2]
                        if score < best_score[i2]:
                            best_score[i2] = score
                            best_edge[i2] = i1
            poses = []
            i1 = str(len(words)-1) + ":" + "</s>"
            while True:
                poses.append(i1.split(":")[1])
                i2 = best_edge[i1]
                i1 = i2
                if "<s>" in i1:
                    poses.append(i1.split(":")[1])
                    break

            poses.reverse()
            poses.remove("<s>")
            poses.remove("</s>")
            print(poses)
            for i in range(0,len(poses)):
                oStr = oStr + poses[i]
                if i == len(poses)-1:
                    oStr += "\n"
                else:
                    oStr += " "

        with open("out.pos", "w") as fo:
            print(oStr, file=fo)


a = 2
if a == 1:
    train("../../test/05-train-input.txt")
    ViterbiDivision("../../test/05-test-input.txt")

else:
    train("../../data/wiki-en-train.norm_pos")
    ViterbiDivision("../../data/wiki-en-test.norm")
    subprocess.call(["perl", "../gradepos.pl","../../data/wiki-en-test.pos","out.pos"])

    # Accuracy: 89.98% (4106/4563)
    # 
    # Most common mistakes:
    # NNS --> NN      45
    # : -->   38
    # NN --> JJ       27
    # NNP --> NN      22
    # JJ --> DT       22
    # JJ --> NN       12
    # VBN --> NN      12
    # NN --> IN       11
    # NN --> DT       10
    # NNP --> JJ      8
