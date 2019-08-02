import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
np.random.seed(42)


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def load(file_name):
    with open(f"./pickles/{file_name}.pkl", 'rb') as f_in:
        data = pickle.load(f_in)
    return data


def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        if f'UNI:{word}' not in ids:
            continue
        phi[ids[f'UNI:{word}']] += 1
    return phi


def predict_one(net, phi_0):
    phi = [0] * (len(net) + 1)
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]
    return 1 if score >= 0 else -1


def test_nn(test_path, out_path):
    net = load('net')
    ids = load('ids')

    with open(out_path, 'w') as f_out:
        for line in open(test_path):
            line = line.rstrip()
            phi = create_features(line, ids)
            predict = predict_one(net, phi)
            print(f'{predict}\t{line}', file=f_out)


if __name__ == '__main__':
    test_path = '../../data/titles-en-test.word'
    out_path = './out.txt'
    test_nn(test_path, out_path)
