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

gModel = None
gPhi = {}こうぎ

def ReLU(x):
    if x < 0:
        return x * 0.0001
    else:
        return x


def dReLU(x):
    if x < 0:
        return 0.0001
    else:
        return 1


class cNN:
    def __init__(self, isA):
        i = 0
        self.A = []
        for k in range(0, len(isA)):
            tL = {}
            for j in range(0, isA[k]):
                tL[j] = i
                i = i + 1
            self.A.append(tL)

        self.x = [0] * i
        self.z = [0] * i
        self.ds_dx = [0] * i
        self.w = {}
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    self.w[str(ip) + ":" + str(jp)] = random.gauss(0, 1)

    def fz(self, k, x):
        if x == len(self.A) - 1:
            return math.tanh(x)
        else:
            if x > 0:
                return x
            else:
                return x*0.001

    def fdz_dx(self, k, x):
        if x == len(self.A) - 1:
            return 1/(math.cosh(x))**2
        else:
            if x > 0:
                return 1
            else:
                return 0.001

    def forward(self, inX):

        # set the input layer.
        for i in self.A[0]:
            x = inX[i]
            ip = self.A[0][i]
            self.x[ip] = x
            self.z[ip] = self.fz(0,x)
        # set the middle layer.
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                x = 0
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    x = x + self.z[jp]* self.w[str(ip) + ":" + str(jp)]
                self.x[ip] = x
                self.z[ip] = self.fz(k, x)
        
        rl = []
        for i in self.A[-1]:
            ip = self.A[-1][i]
            rl.append(self.z[ip])

        return rl

    def backward(self, outZ):

        # set the lastlayer
        for i in self.A[-1]:
            ip = self.A[-1][i]
            y = outZ[i]
            r = y/(1 + self.z[ip]) + (1-y)/2/(1-self.z[ip])
            self.ds_dx[ip] = r * self.fdz_dx(len(self.A)-1, self.x[i])

        for k in reversed(range(1, len(self.A))):
            for j in self.A[k-1]:
                jp = self.A[k-1][j]
                r = 0
                for i in self.A[k]:
                    ip = self.A[k][i]
                    r = r + self.ds_dx[ip] * self.w[str(ip) + ":" + str(jp)]
                self.ds_dx[jp] = r * self.fdz_dx(k, self.x[jp])

        rl = [] 
        for i in self.A[0]:
            ip = self.A[0][i]
            rl.append(self.ds_dx[ip])

        return rl

    def update(self, eps):
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    iw = str(ip) + ":" + str(jp)
                    ds_dw = self.ds_dx[ip]*self.z[jp]
                    self.w[iw] = self.w[iw] + eps*(ds_dw - 0.001*self.w[iw])


def CreateModel(pathInput: str, pathModel: str, N: int):

    gPhi = {}
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
                    float(w)
                    ws2.append("<#>")
                except:
                    ws2.append(w)
            trainSet.append(trainElement(y, ws2))
            for w in ws2:
                if w not in gPhi:
                    gPhi[w] = len(gPhi)

    # fix the number of input variables as the size of the current phi.

    gModel = cNN([len(gPhi), 1])

    # training.
    print("training the model...")
    for i in range(0, N):
        for elem in trainSet:
            y = elem.y
            outZ = [elem.y]
            inX = [0]*len(gPhi)
            
            for w in elem.ws:
                if w in gPhi:
                    inX[gPhi[w]] += 1

            gModel.forward(inX)
            gModel.backward(outZ)
            gModel.update(math.exp(-i*2/N - 2))
        print("current step is ", i)
            

    # out put the model
    #print("outputing the model as ", pathModel, "...")
    #with open(pathModel, "w") as f:
    #    json.dump(weight, f, ensure_ascii=False, indent=4,
    #              sort_keys = True, separators = (',', ': '))


def TestModel(pathInput: str, pathModel: str):
    phi=defaultdict(lambda: 0)
    weight = defaultdict(lambda: 0)
    # load the model.
    #print("loading a model from ", pathModel, "...")
    #with open(pathModel, "r") as f:
    #    for w, v in json.load(f).items():
    #        weight[w]=v
    #        phi[w]=1

    print("testing the model using the data in ", pathInput, "...")
    with open(pathInput, "r") as f:
        with open("answer.labeled", "w") as fo:
            for line in f:
                line=line.strip()
                ws=line.split(" ")

                ws2=[]
                for w in ws:
                    try:
                        j=float(w)
                        ws2.append("<#>")
                    except:
                        ws2.append(w)
                inX = [0]*len(gPhi)
                
                for w in ws2:
                    if w in gPhi:
                        inX[gPhi[w]] += 1

                yp = gModel.forward(inX)
                for w in ws2:
                    yp += weight[w]*phi[w]

                if yp > 0:
                    yp=1
                else:
                    yp=-1

                line=" ".join(ws2)
                print("{}\t{}".format(yp, line), file=fo)


if __name__ == "__main__":

    random.seed(777)
    CreateModel("../../test/03-train-input.txt", "model.txt", 128)
    CreateModel("../../data/titles-en-train.labeled", "modelT.txt", 32)
    TestModel("../../data/titles-en-test.word", "modelT.txt")

    subprocess.call(["../../script/grade-prediction.py",
                     "../../data/titles-en-test.labeled", "answer.labeled"])

    # the output is something like the followings when we use 777 as a seed of random module.
    # $ loading training data from  ../../test/03-train-input.txt ...
    # $ training the model...
    # $ outputing the model as  model.txt ...
    # $ loading training data from  ../../data/titles-en-train.labeled ...
    # $ training the model...
    # $ outputing the model as  modelT.txt ...
    # $ loading a model from  modelT.txt ...
    # $ testing the model using the data in  ../../data/titles-en-test.word ...
    # $ Accuracy = 94.226001%
