import os
from collections import defaultdict
import sys
import math

class nGram:
    prob = 0
    count = 2
    weight = 0
    w_prev = defaultdict(lambda: nGram())

a = nGram
print(a.count)
print(a.w_prev["a"].count)

def fTestUnigram(pathTrain, pathTest):
    myDict = defaultdict(lambda: 0)
    totalCount = 0 
    with open(pathTrain, "r") as f:
        for line in f:
            line = line.strip()
            line += " </s>"
            lSentence = line.split(",")
            for sentence in lSentence: 
                for word in sentence.split(" "):
                    myDict[word] +=1
                    totalCount += 1

            
    coefUnknown = 0.05
    
    word2Prob = defaultdict(lambda: coefUnknown/1000000)
    for key, value in myDict.items():
        word2Prob[key] = value/totalCount * (1-coefUnknown)
    
    oPath = "./" + os.path.split(pathTrain)[1] + ".model"
    with open(oPath, "w") as f:
        for key, value in sorted(word2Prob.items()):
            f.write(key + "," + str(value) + "\n")

    #Calculate entropy and coverage on data
    totalWords = 0
    #calculate empirical distribution
    with open(pathTest, "r") as f:
        for line in f:
            line = line.strip()
            line += " </s>"
            lSentence = line.split(",")
            for sentence in lSentence:
                for word in sentence.split(" "):
                    totalWords +=1
    #calculate entropy and coverage actually
    entropy = 0
    Coverage = 0
    with open(pathTest, "r") as f:
        for line in f:
            line = line.strip()
            line += " </s>"
            lSentence = line.split(",")
            for sentence in lSentence:
                for word in sentence.split(" "):
                    entropy += -1/totalWords * math.log2(word2Prob[word])
                    if word2Prob[word] > coefUnknown/totalCount:
                        Coverage += 1
    Coverage /= totalWords
    print("file: train=", pathTrain, " test=", pathTest)
    print(" number of words: ", totalWords)
    print(" entropy: ", entropy, " Coverage: ", Coverage)

fTestUnigram("../../test/01-train-input.txt", "../../test/01-test-input.txt")
fTestUnigram("../../data/wiki-en-train.word", "../../data/wiki-en-test.word")