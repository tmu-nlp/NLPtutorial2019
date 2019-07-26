import os
from collections import defaultdict
import sys
import math
import json
import random
import collections

#random.seed(777)

weight = {}
bias = {}
sOperation = ["shift", "reduce left", "reduce right"]
for sOp in sOperation:
    weight[sOp] = []
    bias[sOp] = random.gauss(0, 1)
i_phi = {}
pathInput = "../../data/mstparser-en-train.dep"
pathModel = "model.json"
dElements = {}
tdict = {}
tdict["a"] = 0
for i,j in tdict.items():
    print(i,j)

class cElement:
    def __init__(self, index, word, POS, head, label):
        self.index = index
        self.word = word
        self.POS = POS
        self.head = head
        self.children = []
        self.label = label

nEpoch = 256
for epoch in range(0, nEpoch):
    learning_rate = math.exp(-epoch/nEpoch*10)
    print("epoch:",epoch+1,"/",nEpoch, "learning rate:", learning_rate)
    with open(pathInput, "r") as fi:
        queue = collections.deque()
        for line in fi:
            line = line.strip()
            if line == "":
                if len(queue) == 0:
                    break
                # start learning
                stack = collections.deque()
                stack.append(cElement(0, "ROOT", "ROOT", 0, ""))
                unprocessedChild = [0]*(len(queue)+1)
                # init unprocessedChild
                for val in queue:
                    unprocessedChild[val.head] += 1
                stack.append(queue.popleft())
                feats = {}
                while len(queue) > 0 or len(stack) > 1:
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
                        if feat not in i_phi:
                            i_phi[feat] = len(i_phi)
                            for i in sOperation:
                                weight[i].append(random.gauss(0, 1))
                        j = i_phi[feat]
                        j_feats[j] += 1

                    z = defaultdict(lambda:0)
                    zc = defaultdict(lambda:0)
                    zt = defaultdict(lambda:0)

                    Zp = 0

                    for i in sOperation:
                        y_i = 0
                        for j, phi_j in j_feats.items():
                            y_i += weight[i][j]*phi_j + bias[i]
                        zc[i] = math.exp(y_i)
                        Zp += zc[i]
                    for i in sOperation:
                        z[i] = zc[i]/Zp

                    if em1.head == em2.index and unprocessedChild[em1.index] == 0:
                        unprocessedChild[em1.head] -= 1
                        stack.append(em2)
                        zt["reduce right"] = 1
                    elif em2.head == em1.index and unprocessedChild[em2.index] == 0:
                        unprocessedChild[em2.head] -= 1
                        stack.append(em1)
                        zt["reduce left"] = 1
                    else:
                        stack.append(em2)
                        stack.append(em1)
                        stack.append(queue.popleft())
                        zt["shift"] = 1

                    # update weight
                    for i in sOperation:
                        for j, phi_j in j_feats.items():
                            dSdw_ij = -zt[i]*phi_j + z[i]*phi_j
                            weight[i][j] += learning_rate * \
                                (-dSdw_ij - 0.1*weight[i][j])

                        dSdb_i = -zt[i] + z[i]
                        bias[i] += learning_rate*(-dSdb_i)

                continue

            index, word, no, POS, no, no, head, label = line.split("\t")
            queue.append(cElement(int(index), word, POS, int(head), label))

with open(pathModel, "w") as fo:
    dict_weight = {}
    dict_bias = {}
    for i in sOperation:
        dt = {}
        for key, j in i_phi.items():
            dt[str(j)] = weight[i][j]
        dict_weight[i] = dt
        dict_bias[i] = bias[i]

    json.dump({
        "i_phi": i_phi,
        "weight":dict_weight,
        "bias":dict_bias
    },indent=4, sort_keys=True, fp=fo)
