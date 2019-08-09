import os
import sys
import subprocess
from tqdm import tqdm
from itertools import product
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stdout.write("\33[92m" + text + "\33[0m")


def load_model(word_data, tag_data):
    transition = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in zip(word_data, tag_data):
        for t1, t2 in zip(['<s>'] + tags, tags + ['</s>']):
            transition[f'{t1} {t2}'] += 1
        possible_tags.update(tags)
    return dict(transition), possible_tags


def train_hmm_percep(train_path, epoch_num=5):
    """ #12 p26 """
    def update_weight(w, dic, sign):
        for key, value in dic.items():
            w[key] += value * sign

    X, Y_prime = [], []
    for line in open(train_path):
        ws, ps = map(list, zip(*map(lambda x: x.split('_'), line.split())))
        X.append(ws)
        Y_prime.append(ps)
    transition, possible_tags = load_model(X, Y_prime)
    w = defaultdict(int)
    for _ in tqdm(range(epoch_num)):
        for x, y_p in tqdm(zip(X, Y_prime)):
            y_hat = hmm_viterbi(w, x, transition, possible_tags)
            phi_prime = create_feature(x, y_p)
            phi_hat = create_feature(x, y_hat)
            update_weight(w, phi_prime, 1)
            update_weight(w, phi_hat, -1)
    message()
    return w, transition, possible_tags


def create_trans(tag1, tag2):
    """ #12 p27 """
    return [f'T {tag1} {tag2}']


def create_emit(tag, word):
    """ #12 p27 """
    keys = [f'E {tag} {word}']
    if word[0].isupper():
        keys.append(f'CAPS {tag}')
    return keys


def create_feature(x, y):
    """ #12 p28 """
    phi = defaultdict(int)
    for i in range(len(y) + 1):
        first_tag = '<s>' if i == 0 else y[i - 1]
        nxt_tag = y[i] if i != len(y) else '</s>'
        for key in create_trans(first_tag, nxt_tag):
            phi[key] += 1
    for i in range(len(y)):
        for key in create_emit(y[i], x[i]):
            phi[key] += 1
    return phi


def hmm_viterbi(w, words, transition, possible_tags):
    """ #12 p29 """
    def update_best(best_score, best_edge, score, prv, nxt):
        if nxt not in best_score or best_score[nxt] < score:
            best_score[nxt] = score
            best_edge[nxt] = prv

    l = len(words)
    # Forward step
    # BOS
    best_score = {'0 <s>': 0}
    best_edge = {'0 <s>': None}
    # Sequense
    for i, prv, nxt in product(range(l), possible_tags, possible_tags):
        i_prv, prv_nxt = f'{i} {prv}', f'{prv} {nxt}'
        if not (i_prv in best_score and prv_nxt in transition):
            continue
        score = best_score[i_prv]
        score += sum(w[key] for key in create_trans(prv, nxt))
        score += sum(w[key] for key in create_emit(nxt, words[i]))
        update_best(best_score, best_edge, score, i_prv, f'{i + 1} {nxt}')
    # EOS
    for tag in possible_tags:
        l_tag, tag_eos = f'{l} {tag}', f'{tag} </s>'
        if l_tag not in best_score or tag_eos not in transition:
            continue
        score = best_score[l_tag]
        score += sum(w[key] for key in create_trans(tag, '</s>'))
        update_best(best_score, best_edge, score, l_tag, f'{l + 1} </s>')

    # Backward step
    tag_path = []
    nxt_edge = best_edge[f'{l + 1} </s>']
    while nxt_edge != '0 <s>':
        _, tag = nxt_edge.split()
        tag_path.append(tag)
        nxt_edge = best_edge[nxt_edge]
    tag_path.reverse()
    return tag_path


def test_hmm_percep(test_path, out_path, w, transition, tags):
    with open(out_path, 'w') as f:
        for line in open(test_path):
            x = line.split()
            y = hmm_viterbi(w, x, transition, tags)
            print(' '.join(y), file=f)


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        message('[*] test')
        train_path = '../../test/05-train-input.txt'
        test_path = '../../test/05-test-input.txt'
        ans_path = '../../test/05-test-answer.txt'
    else:
        message('[*] main')
        train_path = '../../data/wiki-en-train.norm_pos'
        test_path = '../../data/wiki-en-test.norm'
        ans_path = '../../data/wiki-en-test.pos'

    out_path = './out.txt'
    w, transition, possible_tags = train_hmm_percep(train_path)
    test_hmm_percep(test_path, out_path, w, transition, possible_tags)

    script_path = '../../script/gradepos.pl'
    subprocess.run(f'{script_path} {ans_path} {out_path}'.split())


''' RESULT
[+] CAPS 実装なし
Accuracy: 88.76% (4050/4563)

Most common mistakes:
NNS --> NN      45
JJ --> NN       32
NN --> JJ       29
NN --> NNS      21
NNP --> NN      20
NN --> NNP      18
VBN --> NN      13
VBN --> VBD     13
NNP --> JJ      11
NN --> RB       10

[+] CAPS 実装あり
Accuracy: 86.92% (3966/4563)

Most common mistakes:
NN --> VBG      40
NN --> NNS      33
NN --> VBZ      22
NNS --> NN      21
NN --> JJ       20
JJ --> VBG      20
NN --> NNP      17
JJ --> VBN      15
NNP --> NNS     14
JJ --> JJS      12

[+] tutorial04 (hmm)
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
JJ --> DT       22
NNP --> NN      22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBP --> VB      7

[+] tutorial08 (rnn)
Accuracy: 80.89% (3691/4563)

Most common mistakes:
NN --> NNP      97
JJ --> NNP      47
NNS --> NNP     47
-RRB- --> NNP   43
-LRB- --> NNP   41
VBN --> JJ      34
VBN --> NNP     30
RB --> NNP      28
CD --> VBG      25
NNP --> NN      24
'''
