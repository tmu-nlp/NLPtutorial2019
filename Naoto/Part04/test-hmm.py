import os
import sys
import subprocess
from collections import defaultdict
import itertools
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # cd


def message(text):
    print("\33[92m" + text + "\33[0m")


def model_read(model):
    with open(model, "r") as f:
        transition = defaultdict(float)
        emission = defaultdict(float)
        possible_tags = defaultdict(int)
        for line in f:
            type_, context, word, prob = line.rstrip().split(" ")
            possible_tags[context] = 1
            if type_ == "T":
                transition[f"{context} {word}"] = float(prob)
            else:
                emission[f"{context} {word}"] = float(prob)
    return transition, emission, possible_tags


def foward_back_step(transition: {}, emission: {}, possible_tags: {}, lines: [], path: str):
    with open(path, "w") as f:
        count = 0
        λ = 0.95
        λ_unk = 1 - λ
        N = 100000
        for line in lines:
            words = line.rstrip().split(" ")
            l = len(words)
            best_score = {}
            best_edge = {}
            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = None
            for i in range(l):
                for p, n in itertools.product(possible_tags.keys(), possible_tags.keys()):
                    if not(f"{i} {p}" in best_score and f"{p} {n}" in transition):
                        continue
                    tr = transition[f"{p} {n}"]
                    em = λ * emission[f"{n} {words[i]}"] + λ_unk * (1/N)
                    score = best_score[f"{i} {p}"] + -math.log2(tr) + -math.log2(em)
                    if not(f"{i+1} {n}" in best_score) or best_score[f"{i+1} {n}"] > score:
                        best_score[f"{i+1} {n}"] = score
                        best_edge[f"{i+1} {n}"] = f"{i} {p}"
            for p in possible_tags.keys():
                i = l
                n = "</s>"
                if not(f"{i} {p}" in best_score and f"{p} {n}" in transition):
                    continue
                tr = transition[f"{p} {n}"]
                score = best_score[f"{i} {p}"] + -math.log2(tr)
                if not(f"{i+1} {n}" in best_score) or best_score[f"{i+1} {n}"] > score:
                    best_score[f"{i+1} {n}"] = score
                    best_edge[f"{i+1} {n}"] = f"{i} {p}"
            tags = []
            next_edge = best_edge[f"{l+1} </s>"]
            while next_edge != "0 <s>":
                # このエッジの品詞を出力に追加
                position, tag = next_edge.split(" ")
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            print(" ".join(tags), file=f)


if __name__ == "__main__":
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        model = "05-train-output.txt"
        test = "05-test-input.txt"
        res = "05-test_my_answer.txt"
        ans = "05-test-answer.txt"
    else:
        message("[*] wiki")
        model = "wiki-model.txt"
        test = "wiki-en-test.norm"
        res = "my_answer.pos"
        ans = "wiki-en-test.pos"

    transition, emission, possible_tags = model_read(model)
    with open(test, "r") as f:
        lines = []
        for line in f:
            lines.append(line)
    foward_back_step(transition, emission, possible_tags, lines, res)

    if is_test:
        subprocess.run(f"diff -s {res} {ans}".split())
    else:
        subprocess.run(f"perl gradepos.pl {ans} {res}".split())
    
    message("[+] Done!")

'''
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
JJ --> NN       12
VBN --> NN      12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> VBN      7
'''
