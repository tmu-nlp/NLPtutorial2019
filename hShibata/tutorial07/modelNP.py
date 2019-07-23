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

if False:
    gPhi = {}
    x = np.zeros((5, 4))
    y = np.random.normal(0, 1, (2, 2))
    z = np.full((9), 3)
    w = np.random.normal(0, 1, (1))

    x[1][1] = 1
    print(x)
    print(y)
    print(z)
    print(z*z)
    print(z*w)

    c = 2

    def a():
        global c
        c = 1
    a()
    print(c)

print("dzex:", 0.5/(math.cosh(10)**2), 0.5/(math.cosh(1)**2))


class cNN2:
    def __init__(self, isA, batchSize: int):
        i = 0
        self.A = []
        for k in range(0, len(isA)):
            tL = {}
            for j in range(0, isA[k]):
                tL[j] = i
                i = i + 1
            self.A.append(tL)
        self.batchSize = batchSize
        self.x = np.zeros((i, self.batchSize))
        self.b = np.random.normal(0, 1, (i))
        self.db = np.random.normal(0, 0, (i))
        # for ii in range(0, i):
        #    self.b[ii] = ii
        self.z = np.zeros((i, self.batchSize))
        self.ds_dx = np.zeros((i, self.batchSize))
        self.w = {}
        self.dw = {}
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    self.w[str(ip) + ":" + str(jp)
                           ] = np.random.normal(0, 1, (1))
                    self.dw[str(ip) + ":" + str(jp)
                            ] = np.random.normal(0, 0, (1))

    def fz(self, k, x):
        if k == len(self.A) - 1:
            return (1 + np.tanh(x))/2
        else:
            return np.maximum(x, 0)

    def fdz_dx(self, k,  x):
        if k == len(self.A) - 1:
            return 0.5/(np.cosh(x)**2)
        else:
            return (1+np.sign(x))/2

    def forward(self, inX):

        # set the input layer.
        for i in self.A[0]:
            x = inX[i]
            ip = self.A[0][i]
            self.x[ip] = x
            self.z[ip] = self.fz(0, x)
        # set the middle layer.
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                x = self.b[ip]
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    x = x + self.z[jp] * self.w[str(ip) + ":" + str(jp)]
                self.x[ip] = x
                self.z[ip] = self.fz(k,  x)

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
            r = y/(self.z[ip] + 1e-10) - (1-y)/(1-self.z[ip] + 1e-10)
            self.ds_dx[ip] = -r * self.fdz_dx(len(self.A)-1, self.x[ip])

        for k in reversed(range(1, len(self.A))):
            for j in self.A[k-1]:
                jp = self.A[k-1][j]
                r = 0
                for i in self.A[k]:
                    ip = self.A[k][i]
                    r = r + self.ds_dx[ip] * self.w[str(ip) + ":" + str(jp)]
                self.ds_dx[jp] = r * self.fdz_dx(k,  self.x[jp])

        rl = []
        for i in self.A[0]:
            ip = self.A[0][i]
            rl.append(self.ds_dx[ip])

        return rl

    def show(self):
        for k in range(0, len(self.A)):
            px = []
            pz = []
            pds_dx = []
            pb = []
            pdb = []
            pwi = []
            pdwi = []
            print(len(self.A[k]))
            for i in self.A[k]:
                ip = self.A[k][i]
                px.append(self.x[ip][0])
                pz.append(self.z[ip][0])
                pds_dx.append(self.ds_dx[ip][0])
                pb.append(self.b[ip])
                pdb.append(self.db[ip])
                if k != len(self.A)-1:
                    pwij = []
                    pdwij = []
                    for j in self.A[k+1]:
                        jp = self.A[k+1][j]
                        pwij.append(self.w[str(jp) + ":" + str(ip)])
                        pdwij.append(self.dw[str(jp) + ":" + str(ip)])
                    pwi.append(pwij)
                    pdwi.append(pdwij)
            print("k:", k)
            print("x:", px)
            print("z:", pz)
            print("dsdx:", pds_dx)
            print("b:", pb)
            print("db:", pdb)
            print("w:", pwi)
            print("dw:", pdwi)

    def update(self, eps):
        for k in range(1, len(self.A)):
            for i in self.A[k]:
                ip = self.A[k][i]
                db = eps*(-np.sum(self.ds_dx[ip]))
                self.b[ip] = self.b[ip] + db
                #self.db[ip] = db
                for j in self.A[k-1]:
                    jp = self.A[k-1][j]
                    iw = str(ip) + ":" + str(jp)
                    dw = eps * \
                        (-np.sum(self.ds_dx[ip]*self.z[jp]) - 0.001*self.w[iw])
                    self.w[iw] = self.w[iw] + dw
                    #self.dw[iw] = dw


def CreateModel(pathInput: str, pathModel: str,nLHidden, N: int, Nb: int):
    global gModel
    global gPhi

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

    nLs = [len(gPhi)] + nLHidden + [1]
    # fix the number of input variables as the size of the current phi.
    gModel = cNN2(nLs, Nb)
    # create training dataset on numpy.

    # Make training dataset on numpy.
    vXi = []
    vZo = []
    vbXi = []
    vbZo = []
    for w in gPhi:
        vXi.append(np.zeros((len(trainSet))))
        vbXi.append(np.zeros(Nb))
    vZo.append(np.zeros((len(trainSet))))
    vbZo.append(np.zeros((Nb)))

    for t in range(0, len(trainSet)):
        elem = trainSet[t]
        for w in elem.ws:
            vXi[gPhi[w]][t] += 1
        vZo[0][t] = (elem.y+1)/2
    # training.
    print("training the model...")
   # gModel.show()
    for i in range(0, N):
        print("current epoch:", i+1, "/", N)
        iS = 0
        while iS < len(trainSet):
            print("current sentence:", iS+1, "/", len(trainSet), "epoch:", i+1, "/", N)

            for tp in range(0, Nb):
                t = random.randrange(0, len(trainSet))
                vbZo[0][tp] = vZo[0][t]
                iS += 1
                for j in range(0, len(gPhi)):
                    vbXi[j][tp] = vXi[j][t]

            gModel.forward(vbXi)
            gModel.backward(vbZo)
            gModel.update(math.exp(-i*3/N - 3))

    # out put the model
    #print("outputing the model as ", pathModel, "...")
    # with open(pathModel, "w") as f:
    #    json.dump(weight, f, ensure_ascii=False, indent=4,
    #              sort_keys = True, separators = (',', ': '))


def TestModel(pathInput: str, pathModel: str):
    global gModel
    global gPhi

    vbXi = []
    vbZo = []
    for w in gPhi:
        vbXi.append(np.zeros(gModel.batchSize))
    vbZo.append(np.zeros((gModel.batchSize)))

    print("testing the model using the data in ", pathInput, "...")
    with open(pathInput, "r") as f:
        with open("answer.labeled", "w") as fo:
            i = 0
            Lws2 = []

            def out(i: int, lws):
                yp = gModel.forward(vbXi)
                for j in range(0, i):
                    yp2 = yp[0][j]
                    print(yp2)
                    if yp2 > 0.5:
                        yp2 = 1
                    else:
                        yp2 = -1
                    line = " ".join(lws[j])
                    print("{}\t{}".format(yp2, line))
                    print("{}\t{}".format(yp2, line), file=fo)

            for line in f:
                line = line.strip()
                ws = line.split(" ")

                ws2 = []
                for w in ws:
                    try:
                        float(w)
                        ws2.append("<#>")
                    except:
                        ws2.append(w)
                for j in range(0, len(gPhi)):
                    vbXi[j][i] = 0

                for w in ws2:
                    if w in gPhi:
                        vbXi[gPhi[w]][i] += 1
                Lws2.append(ws2)
                i = i+1
                if i == gModel.batchSize:
                    out(i, Lws2)
                    Lws2 = []
                    i = 0

            out(i, Lws2)


if __name__ == "__main__":
    isTest = False
    random.seed(777)
    np.random.seed(777)
    if isTest:
        CreateModel("test.labeled", "model.txt",[4], 100, 1)
        TestModel("test", "modelT.txt")
        subprocess.call(["../../script/grade-prediction.py",
                         "test.labeled", "answer.labeled"])
    else:
        CreateModel("../../data/titles-en-train.labeled", "modelT.txt",[8], 1, 64)
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
