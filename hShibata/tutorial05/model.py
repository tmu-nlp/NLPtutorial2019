import os
from collections import defaultdict
import sys
import math
import json
import shutil
import subprocess
from random import gauss


def CreateModel(pathInput:str, pathModel:str, N:int):

    phi = defaultdict(lambda: 0)
    weight = defaultdict(lambda: gauss(0, 1))

    trainSet = {}
    print("loading training data from ", pathInput, "...")
    with open(pathInput, "r") as f:
        for line in f:
            tl = line.strip().split("\t")
            y = float(tl[0])
            ws = tl[1].split(" ")
            trainSet[y] = ws
            for w in ws:
                phi[w] = 1


    # training.
    print("training the model...")
    for i in range(0,N):
        for y, ws in trainSet.items():
            yp = 0
            for w in ws:
                yp += weight[w]*phi[w]
            
            if yp > 0:
                yp = 1
            else:
                yp = -1

            weight[w] += 1/2*(y - yp)*weight[w]


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
    
    print("testing the model using the data in ",pathInput ,"...")
    with open(pathInput, "r") as f:
        with open("answer.labeled", "w") as fo:
            for line in f:
                ws = line
                ws = ws.strip().split(" ")
                yp = 0
                for w in ws:
                    yp += weight[w]*phi[w]
                
                if yp > 0:
                    yp = 1
                else:
                    yp = -1

                print(yp,"\t",line, file=fo)
                
        

if __name__ == "__main__":
    CreateModel("../../test/03-train-input.txt", "model.txt", 1000)
    CreateModel("../../data/titles-en-train.labeled", "modelT.txt", 10000)
    TestModel("../../data/titles-en-test.word", "modelT.txt")

    subprocess.call(["../../script/grade-prediction.py", "../../data/titles-en-test.labeled", "answer.labeled"])

