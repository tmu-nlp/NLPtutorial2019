from collections import defaultdict
from tqdm import tqdm
import numpy as np
import dill
import sys
import pickle
# from sklearn.externals import joblib

# sys.setrecursionlimit(100000)


def predict_one(net, phi_0):
    phi = [0 for _ in range(len(net)+1)]
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i+1] = np.tanh(np.dot(w, phi[i])+b).T
    score = phi[len(net)][0][0]
    return 1 if score >= 0 else -1


def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        if f'UNI:{word}' not in ids:
            continue
        phi[ids[f'UNI:{word}']] += 1
    return phi


def test_nn(input_path, output_path):
    with open('net', 'rb') as net_file, open('ids', 'rb') as ids_file:
        net = pickle.load(net_file)
        ids = pickle.load(ids_file)

    with open(input_path, 'r', encoding='utf-8') as i_file, open(output_path, 'w', encoding='utf-8') as o_file:
        for line in i_file:
            phi = create_features(line.strip(), ids)
            predict = predict_one(net, phi)
            print(f'{predict}\t{line.strip()}', file=o_file)


def main():
    input_file_path = '../../data/titles-en-test.word'
    output_file_path = 'result_nn.txt'
    test_nn(input_file_path, output_file_path)


if __name__ == '__main__':
    main()
