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

print("Tutorial 12 model, ver1.0.0")
#pathIn = "test.norm-pos"
pathIn = "../../data/wiki-en-train.norm_pos"
pathModel = "model.json"
prob = defaultdict(lambda: 1e-6)
W_feature = []
W_featureOrg = []
i_phi = {}
setAll = set()
#random.seed(777)
w_decay = 0.00
maxEpoch = 16
LRateOffset = [8,0]
drop_rate = 0.99

a = [1, 2, 3, 4, 5, 6]
print(a[-3:])
print(a[:2])


with open(pathIn, "r") as fi:
    count = defaultdict(lambda: 0)
    nTotal = defaultdict(lambda: 0)
    for line in fi:
        line = line.strip()
        l_w_pos = line.split(" ") + ["</s>_</s>"]
        x_prev = "<s>"
        y_prev = "<s>"
        count["x_0=" + x_prev + "|" + "y_0=" + y_prev] += 1
        for w_pos in l_w_pos:
            x_cur, y_cur = w_pos.split("_")
            x_cur = myEncode.escape(x_cur)
            y_cur = myEncode.escape(y_cur)

            count["y_0=" + y_cur + "|" + "y_-1=" + y_prev] += 1
            count["y=" + y_prev] += 1
            count["x=" + x_prev] += 1
            count["x_0=" + x_cur + "|" + "y_0=" + y_cur] += 1
            lKeyPhi = myEncode.createFeature(x_cur, x_prev, y_cur, y_prev)
            for keyPhi in lKeyPhi:
                if keyPhi not in i_phi:
                    i_phi[keyPhi] = len(i_phi)
                    W_feature.append(random.gauss(0, 0.1))
                    W_featureOrg.append(x_cur)

            nTotal["y"] += 1
            nTotal["x"] += 1

            y_prev = y_cur
            x_prev = x_cur
        count["y=" + y_prev] += 1
        count["x=" + x_prev] += 1

        nTotal["y"] += 1
        nTotal["x"] += 1

    for key, val in count.items():
        keys = key.split("|")
        rand1, value1 = keys[0].split("=")
        if len(keys) == 1:
            prob[key] = count[key]/nTotal[rand1]
        elif len(keys) == 2:
            rand2, value2 = keys[1].split("=")
            rand2_equiv, i = rand2.split("_")
            prob[key] = count[key]/count[rand2_equiv + "=" + value2]

    validState = defaultdict(lambda: set())

    for key, val in prob.items():
        prob[key] = val
        keys = key.split("|")
        rand1, value1 = keys[0].split("=")
        if len(keys) == 2:
            rand2, value2 = keys[1].split("=")
            rand2_equiv, im1 = rand2.split("_")
            rand1_equiv, i0 = rand1.split("_")
            if rand1_equiv == "x":
                validState[rand1_equiv + "=" + value1] |= {value2}
            elif rand1_equiv == "y":
                validState[rand2_equiv + "=" + value2] |= {value1}

            setAll |= {value2}
# training weight of features
#print("aaa", validState)
for epoch in range(0, maxEpoch):
    LearningRate = math.exp(- epoch/maxEpoch*LRateOffset[0]+LRateOffset[1])
    print("cur epoch:", str(epoch+1) + "/" +
          str(maxEpoch), "Learning rate:", LearningRate)
    with open(pathIn, "r") as fi:
        for line in fi:
            line = line.strip()
            l_w_pos = line.split(" ") + ["</s>_</s>"]
            prevBestScore = {"<s>": 0}
            bestEdge = []

            x_prev = "<s>"
            y_prev_t = "<s>"
            for w_pos in l_w_pos:
                x_cur, y_cur_t = w_pos.split("_")
                x_cur = myEncode.escape(x_cur)
                y_cur_t = myEncode.escape(y_cur_t)

                BestScore = {}
                tBestEdge = {"<x_cur>": x_cur, "<x_prev>": x_prev,
                             "<y_cur_t>": y_cur_t, "<y_prev_t>": y_prev_t}
                x_cur = myEncode.escape(x_cur)
                for y_cur in setAll - set({"<s>"}):
                    minLnProb = float("inf")
                    for y_prev, value in prevBestScore.items():
                        feature = 0
                        lKeyPhi = myEncode.createFeature(x_cur, x_prev, y_cur, y_prev)
                        for keyPhi in lKeyPhi:
                            if keyPhi not in i_phi:
                                i_phi[keyPhi] = len(i_phi)
                                W_feature.append(random.gauss(0, 0.1))
                                W_featureOrg.append(x_cur)

                            feature += W_feature[i_phi[keyPhi]]

                        #lnProb = value + \
                        #    -isPT*math.log(prob["y_0=" + y_cur + "|" + "y_-1=" + y_prev]) + \
                        #    -isPE*math.log(prob["x_0=" + x_cur +
                        #                        "|" + "y_0=" + y_cur]) - feature

                        lnProb = value  - feature

                        if lnProb < minLnProb:
                            minLnProb = lnProb
                            tBestEdge[y_cur] = y_prev
                    BestScore[y_cur] = minLnProb
                prevBestScore = BestScore
                bestEdge.append(tBestEdge)
                x_prev = x_cur
                y_prev_t = y_cur_t

            tStr = ""
            y_cur = "</s>"
            for tBestEdge in reversed(bestEdge):
                if y_cur == "<s>":
                    break

                y_prev = tBestEdge[y_cur]
                y_cur_t = tBestEdge["<y_cur_t>"]
                y_prev_t = tBestEdge["<y_prev_t>"]
                x_cur = tBestEdge["<x_cur>"]
                x_prev = tBestEdge["<x_prev>"]
                delta = 1

                lKeyPhi = myEncode.createFeature(x_cur, x_prev, y_cur, y_prev)
                for keyPhi in lKeyPhi:
                    W_feature[i_phi[keyPhi]] -= 1*LearningRate

                lKeyPhi = myEncode.createFeature(x_cur, x_prev, y_cur_t, y_prev_t)
                for keyPhi in lKeyPhi:
                    W_feature[i_phi[keyPhi]] += 1*LearningRate

                y_cur = y_prev
    for key, i in i_phi.items():
        W_feature[i] -= LearningRate*w_decay*W_feature[i]

with open(pathModel, "w") as fo:
    print("output the model...")
    dWi = {}
    for key, i in i_phi.items():
        dWi[key+": " + W_featureOrg[i]] = W_feature[i]

    json.dump({
        "p": prob,
        "W_feature": W_feature,
        "train-set": pathIn,
        "i_phi": i_phi,
        "dWi": dWi
    },
        indent=4, sort_keys=True, fp=fo)

# http://wisdio.com/What-does-mean-such-acronyms-like-NN-NNP-JJ-used-in-POS-part-of-speech-tagging

    #CC - Coordinating conjunction  調整接続詞
    #CD - Cardinal number           基数
    #DT - Determiner                限定詞
    #EX - Existential there
    #FW -Foreign word   
    #IN - Preposition or subordinating conjunction
    #JJ - Adjective                 形容詞
    #JJR - Adjective, comparative   形容詞 比較級
    #JJS - Adjective, superlative   形容詞　最上級
    #LS - List item marker
    #MD - Modal
    #NN - Noun, singular or mass    名詞
    #NNS - Noun, plural             名詞 複数形
    #NNP - Proper noun, singular    固有名詞
    #NNPS - Proper noun, plural     固有名詞　複数形
    #PDT - Predeterminer            
    #POS - Possessive ending        所有格
    #PRP - Personal pronoun         代名詞
    #PRP$ - Possessive pronoun      所有代名詞
    #RB - Adverb                    副詞
    #RBR - Adverb, comparative      副詞　比較級
    #RBS - Adverb, superlative      副詞　最上級
    #RP - Particle
    #SYM - Symbol                   記号
    #TO - to
    #UH - Interjection
    #VB - Verb, base form           動詞　原型
    #VBD - Verb, past tense         動詞　過去形
    #VBG - Verb, gerund or present participle   動名詞？
    #VBN - Verb, past participle                過去分詞
    #VBP - Verb, non-3rd person singular present    複数
    #VBZ - Verb, 3rd person singular present        三人称単数
    #WDT - Wh-determiner
    #WP - Wh-pronoun
    #WP$ - Possessive wh-pronoun
    #WRB - Wh-adverb