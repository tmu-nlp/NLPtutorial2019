import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
from train_rnn import create_one_hot, softmax, find_max, forward_rnn


os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
np.random.seed(42)


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def load(file_name):
    with open(f"./pickles/{file_name}.pkl", 'rb') as f_in:
        data = pickle.load(f_in)
    return data


def test_rnn(test_path, out_path):
    net = load('rnn_net')
    x_ids = load('x_ids')
    y_ids = load('y_ids')

    ids_y = {v: k for k, v in y_ids.items()}
    with open(out_path, 'w') as f_out:
        for line in open(test_path):
            x_list = []
            words = line.split()
            for word in map(lambda x: x.lower(), words):
                if word in x_ids:
                    x_list.append(create_one_hot(x_ids[word], len(x_ids)))
                else:
                    x_list.append(np.zeros((len(x_ids), 1)))
            _, _, y_list = forward_rnn(net, x_list)
            ans = [f'{word}_{ids_y[y]}' for word, y in zip(words, y_list)]
            print(' '.join(ans), file=f_out)


if __name__ == '__main__':
    test_path = '../../data/wiki-en-test.norm'
    out_path = './out.txt'
    test_rnn(test_path, out_path)
