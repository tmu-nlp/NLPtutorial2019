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

print("Tutorial 12, ver1.0.0")
pathInput = "../../test/05-train-input.txt"
pathModel = "model.json"
probPT = {}
probPE = {}
featureT = {}
featureE = {}
with open(pathInput, "r") as fi:
    countPT = defaultdict(lambda: 0)
    countPE = defaultdict(lambda: 0)
    for line in fi:
        line = line.strip()
        l_w_pos = line.split(" ")
        prevWord = "<s>"
        prevPOS = "<s>"
        for w_pos in l_w_pos:
            curWord, curPOS = w_pos.split("_")
            countPT[prevPOS+"->" + curPOS] += 1
            countPE[curPOS+"->" + curWord] += 1
    for key, val in countPT.items():
        probPT[key] = 


with open(pathModel) as fo:
    print("output the model...")
