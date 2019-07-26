import os
from collections import defaultdict
import sys
import subprocess
import json
import math
import collections


def test(pathInput):
    weight = {}
    bias = {}
    sOperation = ["shift", "reduce left", "reduce right"]
    i_phi = {}
    pathModel = "model.json"
    pathOutput = "out.dep"


    class cElement:
        def __init__(self, index, word, POS, head, label):
            self.index = index
            self.word = word
            self.POS = POS
            self.head = head
            self.children = []
            self.label = label


    # at first, import model.
    with open(pathModel, "r") as fm:
        model = json.load(fm)
        i_phi = model["i_phi"]
        for i in sOperation:
            weight[i] = [0]*len(i_phi)
            bias[i] = 0

        for i, val in model["weight"].items():
            for sj, w in val.items():
                j = int(sj)
                weight[i][j] = w
        for i, b in model["bias"].items():
            bias[i] = b

    with open(pathOutput, "w") as fo:
        with open(pathInput, "r") as fi:
            queue = collections.deque()
            t_list = [cElement(0, "ROOT", "ROOT", 0, "ROOT")]
            for line in fi:
                line = line.strip()
                if line == "":
                    if len(queue) == 0:
                        break
                    # start learning
                    stack = collections.deque()
                    stack.append(cElement(0, "ROOT", "ROOT", 0, ""))
                    stack.append(queue.popleft())
                    feats = {}
                    while len(queue) > 0 or len(stack) > 1:
                        if len(stack) <= 1:
                            if len(queue) == 0:
                                break
                            else:
                                stack.append(queue.popleft())
                                continue
                        em1 = stack.pop()
                        em2 = stack.pop()
                        feats = []
                        if len(queue) > 0:
                            e1 = queue[0]
                            feats.append(em1.word + "->" + e1.word)
                            feats.append(em1.POS + "->" + e1.POS)
                            feats.append(em1.POS + "->" + e1.word)
                            feats.append(em1.word + "->" + e1.POS)
                        feats.append(em2.word + "->" + em1.word)
                        feats.append(em2.POS + "->" + em1.POS)
                        feats.append(em2.POS + "->" + em1.word)
                        feats.append(em2.word + "->" + em1.POS)

                        j_feats = defaultdict(lambda: 0)
                        for feat in feats:
                            if feat in i_phi:
                                j = i_phi[feat]
                                j_feats[j] += 1

                        z = defaultdict(lambda: 0)
                        zc = defaultdict(lambda: 0)

                        Zp = 0

                        for i in sOperation:
                            y_i = 0
                            for j, phi_j in j_feats.items():
                                y_i += weight[i][j]*phi_j + bias[i]
                            zc[i] = math.exp(y_i)
                            Zp += zc[i]
                        for i in sOperation:
                            z[i] = zc[i]/Zp

                        if z["shift"] >= z["reduce left"] and z["shift"] >= z["reduce right"] and len(queue) > 0:
                            stack.append(em2)
                            stack.append(em1)
                            stack.append(queue.popleft())
                        elif z["reduce left"] >= z["reduce right"]:
                            t_list[em2.index].head = em1.index
                            stack.append(em1)
                        else:
                            t_list[em1.index].head = em2.index
                            stack.append(em2)
                    for i in range(1, len(t_list)):
                        e_i = t_list[i]
                        print(str(e_i.index) + "\t" + e_i.word + "\t" + e_i.word + "\t" + e_i.POS + "\t" + e_i.POS + "\t_\t" + str(e_i.head) + "\t" + e_i.label, file=fo)
                    print("", file=fo)
                    t_list = [cElement(0, "ROOT", "ROOT", 0, "ROOT")]
                    continue

                index, word, no, POS, no, no, no, label = line.split("\t")
                queue.append(cElement(int(index), word, POS, -1, label))
                t_list.append(cElement(int(index), word, POS, -1, label))

    subprocess.call(["python3", "../../script/grade-dep2.py",pathInput,"out.dep"])

# test accuracy
print("training accuracy")
test("../../data/mstparser-en-train.dep")

print("generalized accuracy")
test("../../data/mstparser-en-test.dep")
