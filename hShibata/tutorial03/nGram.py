import os
from collections import defaultdict
import sys
import math
import json
import shutil



class nGram:
    def __init__(self):
        self.prob = prob_t
        self.count = 0
        self.weight = 0
        self.w_prev = defaultdict(lambda: nGram())

m2Gram = nGram()

def Train(pathCorpus):
    #training 2-gram model using a given corpus.  
    shutil.copy2(pathTrain, os.path.basename(pathTrain))
    with open(pathTrain, "r") as f:
        for line in f:
            line = line.strip()
            sentence = ["<s>"] + line.split(" ") + ["</s>"]
            for i in range(1, len(sentence)):
                w2 = sentence[i]
                w1 = sentence[i-1]
                if w2 not in m2Gram.w_prev:
                    # init myDict[word] as a dictionary.
                    m2Gram.w_prev[w2] = nGram()

                if w1 not in m2Gram.w_prev[w2].w_prev:
                    # init myDict[word] as a dictionary.
                    m2Gram.w_prev[w2].w_prev[w1] = nGram()

                m2Gram.count += 1
                m2Gram.w_prev[w2].count += 1
                m2Gram.w_prev[w2].w_prev[w1].count += 1

            m2Gram.count += 1
            m2Gram.w_prev[sentence[0]].count += 1

    # output a model as json.
    def out(t_nGram):
        oPath = "./" + os.path.split(pathTrain)[1] + ".json"
        with open(oPath, "w") as f:

            w_prev2 = {}

            for w2 in t_nGram.w_prev:
                sw2 = t_nGram.w_prev[w2]
                w_prev1 = {}

                for w1 in sw2.w_prev:
                    sw1 = sw2.w_prev[w1]

                    w_prev1[w1] = {
                        "prob": sw1.prob,
                        "count": sw1.count
                    }

                w_prev2[w2] = {
                    "prob": sw2.prob,
                    "count": sw2.count,
                    "w_prev": w_prev1
                }

            tDict = {"w_prev": w_prev2,
                     "prob": t_nGram.prob,
                     "count": t_nGram.count,
                     }

            json.dump(tDict, f, ensure_ascii=False, indent=4,
                      sort_keys=True, separators=(',', ': '))

    
    m2Gram.prob = 1/1000000
    
    for w2 in c2Gram.w_prev:
        sw2 = m2Gram.w_prev[w2]
        sw2.prob =  r2*sw2.count/m2Gram.count + (1-r2)*m2Gram.prob

        for w1 in sw2.w_prev:
            sw1 = sw2.w_prev[w1]
            sw1.prob = r1*sw1.count/c2Gram.w_prev[w1].count + (1-r1)*sw2.prob
            sw2.w_prev[w1] = sw1

        m2Gram.w_prev[w2] = sw2

    out(m2Gram)

def Prob1(word):
    
def Prob2(word):



def fTestUnigram(pathTrain, pathTest):
    class paramSet:
        Entropy = math.inf
        Coverage = 0
        r1 = 0
        r2 = 0

    bestRes = paramSet()

    # do grid search for r1,r2
    isFirst = True
    for r2 in [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.95]:
        for r1 in [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,0.95]:

            # create a probability model from counts of words.
            m2Gram = createProb(r1,r2)

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
                    prob = 1
                    for i in range(1, len(lW)):
                        if lW[i] in m2Gram.w_prev:
                            Coverage += 1
                            if lW[i-1] in m2Gram.w_prev[lW[i]].w_prev:
                                Entropy += -math.log2(m2Gram.w_prev[lW[i]].w_prev[lW[i-1]].prob)
                            else:
                                Entropy += -math.log2(m2Gram.w_prev[lW[i]].prob*(1-r1))
                        else:
                            Entropy += -math.log2(m2Gram.prob*(1-r2))

                    nWords += len(lW) - 1
                    nSentences += 1

            Coverage /= nWords
            Entropy /= nWords
            print("t: r1 ", r1, " r2 ", r2, " Entropy: ",
                  Entropy, " Coverage: ", Coverage)
            if bestRes.Entropy > Entropy:
                bestRes.Entropy = Entropy
                bestRes.Coverage = Coverage
                bestRes.r1 = r1
                bestRes.r2 = r2

    print("file: train=", pathTrain, " test=", pathTest)
    print(" number of words: ", nWords)
    print(" number of sentences: ", nSentences)
    print(" Entropy: ", bestRes.Entropy, " Coverage: ", bestRes.Coverage)
    print(" r1: ", bestRes.r1, " r2: ", bestRes.r2)


def test():
    nGram.fTestUnigram("../../test/02-train-input.txt", "../../test/02-train-input.txt")    