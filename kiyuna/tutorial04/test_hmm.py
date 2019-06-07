'''
隠れマルコフモデルによる品詞推定
'''
import os
import sys
import subprocess
from collections import defaultdict
from math import log2

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text):
    print("\33[92m" + text + "\33[0m")


def load_model(model_file):
    possible_tags = defaultdict(int)
    emission = defaultdict(float)
    transition = defaultdict(float)
    with open(model_file) as f:
        for line in f:
            type, context, word, prob = line.split()
            possible_tags[context] += 1
            if type == 'T':
                transition[f"{context} {word}"] = float(prob)
            else:
                emission[f"{context} {word}"] = float(prob)
    return possible_tags, emission, transition


def test_hmm(model_path, test_path, output_path):
    λ_1 = 0.90
    λ_unk = 1 - λ_1
    V = 1e6

    possible_tags, emission, transition = load_model(model_path)

    res = []
    with open(test_path) as f:
        for line in f:
            words = line.split()

            # 最小化DP（viterbi）
            best_score = defaultdict(lambda: float('inf'))
            best_edge = defaultdict(str)

            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = None

            for i, word in enumerate(words):
                for prev in possible_tags:
                    for next in possible_tags:
                        if f"{i} {prev}" not in best_score:
                            continue
                        if f"{prev} {next}" not in transition:
                            continue

                        score = best_score[f"{i} {prev}"]
                        Pt = transition[f"{prev} {next}"]
                        score += -log2(Pt)
                        Pe = λ_1 * emission[f"{next} {word}"] + λ_unk / V
                        score += -log2(Pe)

                        if best_score[f"{i+1} {next}"] > score:
                            best_score[f"{i+1} {next}"] = score
                            best_edge[f"{i+1} {next}"] = f"{i} {prev}"

            l = len(words)
            for tag in possible_tags:
                if f"{l} {tag}" not in best_score:
                    continue
                if f"{tag} </s>" not in transition:
                    continue

                Pt = transition[f"{tag} </s>"]
                score = best_score[f"{l} {tag}"] + -log2(Pt)

                if best_score[f"{l+1} </s>"] > score:
                    best_score[f"{l+1} </s>"] = score
                    best_edge[f"{l+1} </s>"] = f"{l} {tag}"

            tags = []
            next_edge = best_edge[f"{l+1} </s>"]
            while next_edge != "0 <s>":
                pos, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]

            tags.reverse()
            res.append(" ".join(tags) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(res)


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        model = './model_test.txt'
        test = '../../test/05-test-input.txt'
        res = './result_test.pos'
        ans = '../../test/05-test-answer.txt'
    else:
        message("[*] wiki")
        model = './model_wiki.txt'
        test = '../../data/wiki-en-test.norm'
        res = './result_wiki.pos'
        ans = '../../data/wiki-en-test.pos'

    test_hmm(model, test, res)

    if is_test:
        subprocess.run(f'diff -s {res} {ans}'.split())
    else:
        subprocess.run(f'perl ../../script/gradepos.pl {ans} {res}'.split())

    message("[+] Done!")


'''
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
'''
