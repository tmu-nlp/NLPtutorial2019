import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm


def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', help='学習用ファイル', type=str)
    parser.add_argument('input_file', help='入力ファイル', type=str)
    return parser.parse_args()


def load_train_data(train_path):
    with open(train_path, 'r') as t_file:
        for line in t_file:
            elements = line.rstrip('\n').split()
            X = []
            Y = []
            for element in elements:
                word, tag = element.split('_')
                X.append(word)
                Y.append(tag)
            yield X, Y


def load_test_data(test_path):
    with open(test_path, 'r') as i_file:
        for line in i_file:
            X = line.strip('\n').split()
            yield X


def create_trans(tag1, tag2):
    return [f'T,{tag1},{tag2}']


def create_emit(tag, word):
    result = [f'E,{tag},{word}']
    if word[0].isupper():
        result.append(f'CAPS,{tag}')
    return result


def hmm_viterbi(w, X, transition, tags):
    l = len(X)
    best_score = {'0 <s>': 0}
    best_edge = {'0 <s>': None}

    for i in range(l):
        for prev in tags:
            for next_ in tags:
                if f'{i} {prev}' not in best_score or f'{prev} {next_}' not in transition:
                    continue
                score = best_score[f'{i} {prev}'] + sum(
                    w[key] for key in create_trans(prev, next_) + create_emit(next_, X[i]))
                if f'{i+1} {next_}' not in best_score or best_score[f'{i+1} {next_}'] < score:
                    best_score[f'{i+1} {next_}'] = score
                    best_edge[f'{i+1} {next_}'] = f'{i} {prev}'

    for tag in tags:
        if f'{tag} </s>' not in transition:
            continue
        score = best_score[f'{l} {tag}'] + \
            sum(w[key] for key in create_trans(tag, '</s>'))
        if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] < score:
            best_score[f'{l+1} </s>'] = score
            best_edge[f'{l+1} </s>'] = f'{l} {tag}'

    tag_path = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        _, tag = next_edge.split()
        tag_path.append(tag)
        next_edge = best_edge[next_edge]
    tag_path.reverse()

    return tag_path


def create_features(X, Y):
    phi = defaultdict(int)
    Y_ = ['<s>'] + Y + ['</s>']
    for i in range(len(Y_)-1):
        first_tag = Y_[i]
        next_tag = Y_[i+1]
        for key in create_trans(first_tag, next_tag):
            phi[key] += 1
        if i == len(Y_)-2:
            break
        for key in create_emit(Y[i], X[i]):
            phi[key] += 1

    return phi


def train_hmm_percep(train_path, epoch=5):
    transition = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in load_train_data(train_path):
        tags_ = ['<s>'] + tags + ['</s>']
        for i in range(len(tags_)-1):
            transition[f'{tags_[i]} {tags_[i+1]}'] += 1
        possible_tags.update(tags)

    w = defaultdict(int)
    for _ in tqdm(range(epoch)):
        for X, Y_prime in load_train_data(train_path):
            Y_hat = hmm_viterbi(w, X, transition, possible_tags)
            phi_prime = create_features(X, Y_prime)
            phi_hat = create_features(X, Y_hat)
            for key, value in phi_prime.items():
                w[key] += value
            for key, value in phi_hat.items():
                w[key] -= value

    with open('hmm_percep.model', 'wb') as m_file:
        pickle.dump((dict(transition), possible_tags, w), m_file)


def test_hmm_percep(test_path):
    with open('hmm_percep.model', 'rb') as m_file:
        transition, possible_tags, w = pickle.load(m_file)

    with open('answer.txt', 'w') as o_file:
        for X in load_test_data(test_path):
            Y_hat = hmm_viterbi(w, X, transition, possible_tags)
            print(' '.join(Y_hat), file=o_file)


def main():
    args = arguments_parse()
    train_path = args.train_file
    test_path = args.input_file
    train_hmm_percep(train_path)
    test_hmm_percep(test_path)


if __name__ == '__main__':
    main()

# Accuracy: 88.76% (4050/4563)
# Most common mistakes:
# NNS --> NN      33
# NN --> JJ       31
# JJ --> VBN      31
# NN --> NNS      24
# NN --> NNP      18
# VBN --> NNS     15
# NN --> VBN      13
# JJ --> NN       13
# JJ --> RB       12
# NNS --> JJ      12
