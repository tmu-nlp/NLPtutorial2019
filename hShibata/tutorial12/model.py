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
pathInput = "../../test/05-train-input"
pathModel = "model.json"
probPT = {}
probPE = {}
featureT = {}
featureE = {}
with open(pathInput, "r") as fi:
    for line in fi:
        line = line.strip()
        l_w_pos = line.split(" ")
        for w_pos in l_w_pos:
            word, POS = w_pos.split("_")
            print(word, POS)


with open(pathModel) as fo:
    print("output the model...")