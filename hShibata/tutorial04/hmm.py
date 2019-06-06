import subprocess
import shutil
import json
import math
import os
from collections import defaultdict
import sys
import nGram # generalized n-gram module. in this program, this is used for the hidden variables.


# a class contains count of words, probability, and smoothing value (weight). this class will be constructed recursively to denote conditional distribution.
class c_PE:
    def __init__(self, weight):
        self.count = 0
        self.prob = 0
        self.weight = weight
        self.pRand = defaultdict(lambda: c_PE(weight))

# grobal variables.
gPE_given_pos = c_PE(0.95) # this value in constructor denotes the smoothing value of uni-gram.
gN = 10**6 # assumed number of vocabulary in the real world.
gPosSet = set() 

def train(path):
    # this function trains the model, in which conditional probability of the state of word given the state of Part of Speech (POS), and transition probability of hidden units are estimated.

    # path: file path to a training data.
    strPos = "" # variable for storing pos tagging data to train 2-gram of Hidden units.
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
                gPosSet.add(pos) # add also to a state set of POS tags. 

    # build a bi-gram model for the POS Markov chain.
    with open("tempPOS.word", "w") as f:
        print(strPos[:-1], file=f)

    nGram.mnGram = nGram.c_nGram(1)  # 1 means no smoothing method is used.
    nGram.gN = 2  # 2 means bi-gram.
    nGram.Train("tempPOS.word") # data set file of tag data extracted in the above.

    # build a uni-gram model for the visible variable probability.
    for pos, gvn_pos in gPE_given_pos.pRand.items():
        for word, w_gvn_pos in gvn_pos.pRand.items():
            w_gvn_pos.prob = w_gvn_pos.count/gvn_pos.count


def probW(word, pos):
    # this function estimate the conditional probability of a word given a POS value.
    if pos not in gPE_given_pos.pRand:
        raise Exception("unknown pos was found. this model is not supported.") # should not occur.

    gvn_pos = gPE_given_pos.pRand[pos]
    weight = gPE_given_pos.weight
    if word in gvn_pos.pRand:
        w_gvn_pos = gvn_pos.pRand[word]
        return weight*w_gvn_pos.prob + (1-weight)*1/gN
    else:
        return (1-weight)*1/gN


def ViterbiDivision(path):
    # this function runs Viterbi algorithm on the given graph, which defined as Graph in the following.
    with open(path, "r") as f:
        oStr = ""
        for line in f:
            line = line.strip()
            words = ["<s>"] + line.split(" ") + ["</s>"]
            Graph = {} # graph Viterbi on which algorithm will run.

            # create a graph for a sentence.
            for i in range(1, len(words)):

                # restrict the state set of the start and end of this sentence to "<s>" and "</s>" respectively.

                # 1 means previous state, which is denoted using nodes of a graph. 2 means the current state.
                if i == 1:
                    posSet1 = set({"<s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"}) # remove the end and start tags from an ordinal state set, which are neightier the start nor the end.
                elif i == len(words)-1:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = set({"</s>"})
                else:
                    posSet1 = gPosSet - set({"<s>","</s>"})
                    posSet2 = gPosSet - set({"<s>","</s>"})

                word = words[i] # get a word for index i.
                for pos1 in posSet1:
                    tG = {}
                    for pos2 in posSet2:
                        # estimate a log likelihood on the posterior distribution of POS tags, given the word.
                        ln = - math.log(nGram.probWord([pos1,pos2])+0.1**100) - math.log(probW(word, pos2))

                        tG[str(i) + ":" +  pos2] = ln # make an index to be able to represent whole nodes enough.

                    Graph[str(i-1) + ":" +  pos1] = tG

            # Run Viterbi algorithm.
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
                    i2 = str(i) + ":" + pos2 # reconstruct the index corresponding to the above ones.
                    best_score[i2] = math.inf # initialize with the highest cost.
                    for pos1 in posSet1:
                        i1 = str(i-1)  + ":" +  pos1
                        score = best_score[i1] + Graph[i1][i2]
                        if score < best_score[i2]:
                            best_score[i2] = score
                            best_edge[i2] = i1

            poses = [] # this variable will contain the sequence of POS tags which show the highest likelihood given the sequence of words.

            i1 = str(len(words)-1) + ":" + "</s>"
            while True:
                poses.append(i1.split(":")[1]) # retrive tags from the index.
                i2 = best_edge[i1]
                i1 = i2
                if "<s>" in i1:
                    break

            poses.reverse()

            # we have to make some addjustmet to evaluate this model using gradepos.pl
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
