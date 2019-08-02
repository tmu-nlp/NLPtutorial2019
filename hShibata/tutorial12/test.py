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
import myEncode

def main():
    
    print("Tutorial 12 test, ver1.0.0")

    def A():
        x = 1

        def B():
            nonlocal x
            x = 2
        B()
        print(x)
    A()

    pathIn = "test.norm"
    pathIn = "../../data/wiki-en-test.norm"
    #pathIn = "../../data/wiki-en-train.norm"
    pathTestIn = "test.norm-pos"
    pathTestIn = "../../data/wiki-en-test.norm_pos"
    #pathTestIn = "../../data/wiki-en-train.norm_pos"
    pathModel = "model.json"
    pathOut = "out.pos"
    prob = defaultdict(lambda: 1e-6)

    validState = defaultdict(lambda: set())
    W_feature = {}
    i_phi = {}

    with open(pathModel, "r") as fm:
        tDict = json.load(fm)
        W_feature = tDict["W_feature"]
        i_phi = tDict["i_phi"]
        setAll = set()

        for key, val in tDict["p"].items():
            prob[key] = val
            keys = key.split("|")
            rand1, value1 = keys[0].split("=")
            if len(keys) == 2:
                rand2, value2 = keys[1].split("=")
                rand2_equiv, im1 = rand2.split("_")
                rand1_equiv, i0 = rand1.split("_")
                # print(rand1_equiv + "=" + value1 +
                #      "|" + rand2_equiv + "=" + value2)
                if rand1_equiv == "x":
                    validState[rand1_equiv + "=" + value1] |= {value2}
                elif rand1_equiv == "y":
                    validState[rand2_equiv + "=" + value2] |= {value1}

                setAll |= {value2}

        # print(validState)
        validState.default_factory = lambda: setAll
        # traing weight of features

    print(i_phi)
    print(W_feature)
    with open(pathOut, "w") as fo:
        with open(pathIn, "r") as fi:
            for line in fi:
                line = line.strip()

                bestEdge = []

                l_w = line.split(" ") + ["</s>"]
                prevBestScore = {"<s>": 0}
                x_prev = "<s>"
                for x_cur in l_w:
                    BestScore = {}
                    tBestEdge = {"<x_cur>": x_cur}
                    x_cur = myEncode.escape(x_cur)
                    for y_cur in validState["x=" + x_cur]:
                        minLnProb = float("inf")
                        #print(prevBestScore, y_cur, x_cur)
                        for y_prev, value in prevBestScore.items():
                            feature = 0
                            lKeyPhi = myEncode.createFeature(x_cur, x_prev, y_cur, y_prev)
                            for keyPhi in lKeyPhi:
                                if keyPhi in i_phi:
                                    feature += W_feature[i_phi[keyPhi]]
                                    #print(keyPhi,W_feature[i_phi[keyPhi]])
                            #feature = 0

                            lnProb = value + \
                                -math.log(prob["y_0=" + y_cur + "|" + "y_-1=" + y_prev]) + \
                                -math.log(prob["x_0=" + x_cur +
                                               "|" + "y_0=" + y_cur]) - feature
                            if lnProb < minLnProb:
                                minLnProb = lnProb
                                tBestEdge[y_cur] = y_prev
                        BestScore[y_cur] = minLnProb
                    prevBestScore = BestScore
                    bestEdge.append(tBestEdge)
                    x_prev = x_cur

                tStr = ""
                y_cur = "</s>"
                for tBestEdge in reversed(bestEdge):
                    if y_cur == "<s>":
                        break
                    elif y_cur != "</s>":
                        if tStr != "":
                            tStr = " " + tStr
                        tStr = tBestEdge["<x_cur>"] + "_" + y_cur + tStr
                    y_cur = tBestEdge[y_cur]

                print(line)
                print(tStr, file=fo)
    print("testing using test data set...")
    subprocess.call(["perl", "../gradepos.pl", pathTestIn, "out.pos"])


main()
