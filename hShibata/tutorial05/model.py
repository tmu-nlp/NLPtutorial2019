import os
from collections import defaultdict
import sys
import math
import json
import shutil
import subprocess
import random 
import operator


def CreateModel(pathInput:str, pathModel:str, N:int):

    phi = defaultdict(lambda: 0)
    weight = defaultdict(lambda: random.gauss(0, 1))

    trainSet = []
    class trainElement:
        def __init__(self, y, ws):
            self.y = y
            self.ws = ws

    print("loading training data from ", pathInput, "...")
    with open(pathInput, "r") as f:
        for line in f:
            tl = line.strip().split("\t")
            y = float(tl[0])
            ws = tl[1].split(" ")
            ws2 = []
            for w in ws:
                try:
                    j = float(w)
                    ws2.append("<#>")
                except:
                    ws2.append(w)
            trainSet.append(trainElement(y, ws2))
            for w in ws2:
                phi[w] = 1


    # training.
    print("training the model...")
    for i in range(0,N):
        for elem in trainSet: 
            y = elem.y
            yp = 0
            for w in elem.ws:
                yp += weight[w]*phi[w]
            
            if yp > 0:
                yp = 1
            else:
                yp = -1

            for w in elem.ws:
                weight[w] += math.exp(-i*2/N )*(y - yp)*phi[w]

    # out put the model
    print("outputing the model as ", pathModel, "...")
    with open(pathModel, "w") as f:
        json.dump(weight, f, ensure_ascii=False, indent=4,
                    sort_keys=True, separators=(',', ': '))


def TestModel(pathInput: str, pathModel:str):
    phi = defaultdict(lambda: 0)
    weight = defaultdict(lambda: 0)
    # load the model.
    print("loading a model from ", pathModel, "...")
    with open(pathModel, "r") as f:
        for w, v in json.load(f).items():
            weight[w] = v
            phi[w] = 1
    
    print("testing the model using the data in ",pathInput ,"...")
    with open(pathInput, "r") as f:
        with open("answer.labeled", "w") as fo:
            for line in f:
                line = line.strip()
                ws = line.split(" ")

                ws2 = []
                for w in ws:
                    try:
                        j = float(w)
                        ws2.append("<#>")
                    except:
                        ws2.append(w)
                yp = 0
                for w in ws2:
                    yp += weight[w]*phi[w]
                
                if yp > 0:
                    yp = 1
                else:
                    yp = -1

                line = " ".join(ws2)
                print("{}\t{}".format(yp,line), file=fo)
                
        

if __name__ == "__main__":

    random.seed(777)
    CreateModel("../../test/03-train-input.txt", "model.txt", 128)
    CreateModel("../../data/titles-en-train.labeled", "modelT.txt", 32)
    TestModel("../../data/titles-en-test.word", "modelT.txt")

    subprocess.call(["../../script/grade-prediction.py", "../../data/titles-en-test.labeled", "answer.labeled"])

    # the output is something like the followings when we use 777 as a seed of random module.
    #$ loading training data from  ../../test/03-train-input.txt ...
    #$ training the model...
    #$ outputing the model as  model.txt ...
    #$ loading training data from  ../../data/titles-en-train.labeled ...
    #$ training the model...
    #$ outputing the model as  modelT.txt ...
    #$ loading a model from  modelT.txt ...
    #$ testing the model using the data in  ../../data/titles-en-test.word ...
    #$ Accuracy = 94.226001%
    