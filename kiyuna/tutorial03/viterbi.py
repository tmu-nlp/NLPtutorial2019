'''
形態素で分割するプログラム
'''
import os
import sys
import math
import subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def n_gram(seq, n):
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def load_model(path):
    probs = defaultdict(float)
    with open(path) as f:
        for line in f:
            word, prob = line.split('\t')
            probs[word] = float(prob)
    return probs


def viterbi(P_uni, input, output):
    λ_1 = 0.95
    λ_unk = 1 - λ_1     # 未知語確率
    V = 1000000         # 未知語を含む語彙数

    res = []
    with open(input) as f_in:
        # Forward step
        for line in map(lambda x: x.strip(), f_in):
            size = len(line)
            best_edge = [None] * (size + 1)
            best_score = [float('inf')] * (size + 1)
            best_score[0] = 0
            for word_end in range(1, size + 1):
                for word_begin in range(size):
                    word = line[word_begin:word_end]
                    if word in P_uni or len(word) == 1:
                        prob = λ_1 * P_uni[word] + λ_unk / V
                        my_score = best_score[word_begin] + -math.log2(prob)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)
            # Backward step
            words = []
            next_edge = best_edge[-1]
            while next_edge:
                words.append(line[next_edge[0]:next_edge[1]])
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            res.append(' '.join(words) + '\n')
    with open(output, 'w') as f_out:
        f_out.writelines(res)
    return


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        path_m = '../../test/04-model.txt'
        input = '../../test/04-input.txt'
        output = './res_test.word'
    else:
        message("[*] wiki")
        path_m = './model_wiki.txt'
        input = '../../data/wiki-ja-test.txt'
        output = './res_wiki.word'

    model = load_model(path_m)
    viterbi(model, input, output)

    if is_test:
        subprocess.run(
            f'diff -s {output} ../../test/04-answer.txt'.split())

    # 分割精度の評価
    subprocess.run(
        f'perl ../../script/gradews.pl\
        ../../data/wiki-ja-test.word {output}'.split())

    message("[+] Finished!")


''' RESULT
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
'''
