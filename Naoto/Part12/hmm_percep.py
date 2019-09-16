import os
import subprocess
from collections import defaultdict
from tqdm import tqdm
from itertools import product
import sys


def message(text="", CR=False):
    text = '\r' + text if CR else text + "\n"
    sys.stdout.write("\33[92m" + text + "\33[0m")


def create_trans(tag1, tag2):
    """ #12 p27 """
    return [f'T {tag1} {tag2}']


def create_emit(tag, word):
    """ #12 p27 """
    keys = [f'E {tag} {word}']
    if word[0].isupper():
        keys.append(f'CAPS {tag}')
    return keys


def load_model(word_data, tag_data):
    transition = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in zip(word_data, tag_data):
        for t1, t2 in zip(['<s>'] + tags, tags + ['</s>']):
            transition[f'{t1} {t2}'] += 1
        possible_tags.update(tags)
    return dict(transition), possible_tags


def create_features(X, Y):
    """ #12 p28 """
    phi = defaultdict(int)
    for i in range(len(Y) + 1):
        first_tag = '<s>' if i == 0 else Y[i - 1]
        nxt_tag = '</s>' if i == len(Y) else Y[i]
        for key in create_trans(first_tag, nxt_tag):
            phi[key] += 1
    for i in range(len(Y)):
        for key in create_emit(Y[i], X[i]):
            phi[key] += 1
    return phi


def hmm_viterbi(w, X, transition, possible_tags):
    """ #12 29 """
    def update_best(best_score, best_edge, score, prv, nxt):
        if nxt not in best_score or best_score[nxt] < score:
            best_score[nxt] = score
            best_edge[nxt] = prv

    # Foward step
    # BOS
    best_score = {'0 <s>': 0}
    best_edge = {'0 <s>': None}
    l = len(X)
    # Sequence
    for i, prv, nxt in product(range(l), possible_tags, possible_tags):
        i_prv, prv_nxt = f'{i} {prv}', f'{prv} {nxt}'
        if not (i_prv in best_score and prv_nxt in transition):
            continue
        score = best_score[i_prv]
        score += sum(w[key] for key in create_trans(prv, nxt))
        score += sum(w[key] for key in create_emit(nxt, X[i]))
        update_best(best_score, best_edge, score, i_prv, f'{i + 1} {nxt}')
    # EOS
    for tag in possible_tags:
        l_tag, tag_eos = f'{l} {tag}', f'{tag} </s>'
        if l_tag not in best_score or tag_eos not in transition:
            continue
        score = best_score[l_tag]
        score += sum(w[key] for key in create_trans(l_tag, tag_eos))
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


def train_hmm_percep(train_path, epoch_num=5):
    """ #12 26 """
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
            Y_hat = hmm_viterbi(w, x, transition, possible_tags)
            phi_prime = create_features(x, y_p)
            phi_hat = create_features(x, Y_hat)
            update_weight(w, phi_prime, 1)
            update_weight(w, phi_hat, -1)
    return w, transition, possible_tags


def test_hmm_percep(test_path, out_path, w, transition, tags):
    with open(out_path, 'w') as f_out:
        for line in open(test_path):
            x = line.split()
            y = hmm_viterbi(w, x, transition, tags)
            print(' '.join(y), file=f_out)


def main():
    if sys.argv[1:] == ["test"]:
        message('[*] test')
        input_path = '../../../nlptutorial/test/05-train-input.txt'
        test_path = '../../../nlptutorial/test/05-test-input.txt'
        ans_path = '../../../nlptutorial/test/05-test-answer.txt'
    else:
        message('[*] main')
        input_path = '../../../nlptutorial/data/wiki-en-train.norm_pos'
        test_path = '../../../nlptutorial/data/wiki-en-test.norm'
        ans_path = '../../../nlptutorial/data/wiki-en-test.pos'

    out_path = 'out.txt'
    w, transition, possible_tags = train_hmm_percep(input_path)
    test_hmm_percep(test_path, out_path, w, transition, possible_tags)

    script_path = '../../../nlptutorial/script/gradepos.pl'
    subprocess.run(f'perl {script_path} {ans_path} {out_path}'.split())


if __name__ == '__main__':
    main()
