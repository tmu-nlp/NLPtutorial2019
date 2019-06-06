import os
from collections import defaultdict
import sys
import math
import json
import shutil

gDecRate = 4

# nGram is a class to describe components in which nGram is described. this class is used recursively.
class c_nGram:
    def __init__(self, weight: float, depth=0):
        self.prob = 0
        self.count = 0
        self.weight = weight
        self.depth = depth
        self.w_prev = defaultdict(lambda: c_nGram(math.pow(weight, gDecRate), depth+1))


pathsTrain = []
# global variable to denote n gram actually.
mnGram = c_nGram(0.98)
gN = 1
nVirtualWords = 1000000

def Train(pathCorpus: str):
    # training n-gram model using a given corpus.  N means the n of n-gram.
    pathsTrain.append(pathCorpus)
    with open(pathCorpus, "r") as f:
        for line in f:
            line = line.strip()
            sentence = ["<s>"] + line.split(" ") + ["</s>"]

            # currently we will use tri gram.

            # function counts nGram recursively until depth becomes 0.
            def recCount(itGram: c_nGram, it: int):
                itGram.count += 1

                if it == -1 or itGram.depth == gN:
                    # the last recursive call.
                    return
                else:
                    w = sentence[it]
                    # We should be able to change the member values of arguments if it is an instance of a class in python3
                    recCount(itGram.w_prev[w], it-1)

            for i in range(0, len(sentence)):
                recCount(mnGram, i)

    mnGram.prob = 1/nVirtualWords
    # Setting probability.

    def recProbBuilder(itGram: c_nGram):
        for key, val in itGram.w_prev.items():
            val.prob = val.count/itGram.count
            recProbBuilder(val)

    recProbBuilder(mnGram)

    # output a model as json.
    oPath = "./myNGram.json"
    with open(oPath, "w") as f:
        def recOut(itGram: c_nGram):
            tJson = {}
            for key, val in itGram.w_prev.items():
                tJson[key] = recOut(val)

            return {"w_prev": tJson,
                    "prob": itGram.prob,
                    "count": itGram.count,
                    "weight": itGram.weight,
                    "depth": itGram.depth,
                    }


        oJson = recOut(mnGram)
        oJson["paths_to_train_data"] = pathsTrain
        json.dump(oJson, f, ensure_ascii=False, indent=4,
                  sort_keys=True, separators=(',', ': '))

def lnProb(sentence):
    S = 0
    for i in range(1, len(sentence)):
        def recProb(itGram: c_nGram, prob: float, it: int):
            prob = itGram.weight*itGram.prob + (1-itGram.weight)*prob
            w = sentence[it]
            if it == 0 or itGram.depth == gN:
                return prob
            else:
                return recProb(itGram.w_prev[w], prob, it-1)

        S += math.log2(recProb(mnGram, 0, i))
    return S
def probWord(words):
    tw = words
    def recProb(itGram: c_nGram, prob: float, it: int):
        prob = itGram.weight*itGram.prob + (1-itGram.weight)*prob
        if it == -1 or itGram.depth == len(tw):
            return prob
        else:
            w = tw[it]
            return recProb(itGram.w_prev[w], prob, it-1)
    return recProb(mnGram, 0, len(tw)-1)

def fTestUnigram(pathTest):

    # do grid search for r1,r2

    # Calculate entropy and coverage on data
    # Entropy should be calculated for each sentence, not 2 gram.
    nSentences = 0
    nWords = 0

    # calculate entropy and coverage actually
    Entropy = 0
    Coverage = 0
    with open(pathTest, "r") as f:
        for line in f:
            line = line.strip()
            lW = ["<s>"] + line.split(" ") + ["</s>"]
            Entropy += -lnProb(lW)
            nWords += len(lW) - 1
            nSentences += 1
            for i in range(1,len(lW)):
                if probWord([lW[i]]) > mnGram.prob:
                    Coverage += 1

    Coverage /= nWords
    Entropy /= nWords
    print(" number of words: ", nWords)
    print(" number of sentences: ", nSentences)
    print(" Entropy: ", Entropy, " Coverage: ", Coverage)

if __name__ == "__main__":
    path_this = os.path.dirname(os.path.realpath(__file__))

    mnGram = c_nGram(0.98)
    gN = 2
    nVirtualWords = 1000000
    gDecRate = 8
    #Train(path_this + "/../../test/02-train-input.txt")
    #fTestUnigram(path_this + "/../../test/02-train-input.txt")

    Train(path_this + "/../../data/wiki-en-train.word")
    fTestUnigram(path_this + "/../../data/wiki-en-test.word")