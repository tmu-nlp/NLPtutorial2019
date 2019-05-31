from collections import defaultdict
from math import log2
import re
import sys


def main(filename: str):
    # 定数
    unk = 0.05
    V = 1e6

    possible_tags = defaultdict(int)
    emission = defaultdict(int)
    transition = defaultdict(int)

    # モデルの読み込み
    for line in open("model.txt", "r", encoding="utf-8"):
        t, context, word, prob = re.split("[\t\s]", line.strip())
        possible_tags[context] += 1
        if t == "T":
            transition[f"{context} {word}"] = float(prob)
        else:
            emission[f"{context} {word}"] = float(prob)

    for line in open(filename, "r", encoding="utf-8"):
        words = line.rstrip().split()
        l = len(words)

        best_score = defaultdict(float)
        best_edge = defaultdict(str)

        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None

        tags = possible_tags.keys()
        for i in range(l):
            for prev in tags:
                for next in tags:
                    if (f"{i} {prev}" in best_score) and (f"{prev} {next}" in transition):
                        p_t = transition[f"{prev} {next}"]
                        p_e = (1 - unk) * emission[f"{next} {words[i]}"] + unk / V
                        score = best_score[f"{i} {prev}"] - log2(p_t) - log2(p_e)

                        if f"{i+1} {next}" not in best_score or best_score[f"{i+1} {next}"] > score:
                            best_score[f"{i+1} {next}"] = score
                            best_edge[f"{i+1} {next}"] = f"{i} {prev}"

        for tag in tags:
            if f"{tag} </s>" in transition:
                p_t = transition[f"{tag} </s>"]
                score = best_score[f"{l} {tag}"] - log2(p_t)
                if (f"{l+1} </s>" not in best_score or best_score[f"{l+1} </s>"] > score):
                    best_score[f"{l+1} </s>"] = score
                    best_edge[f"{l+1} </s>"] = f"{l} {tag}"

        tags = []
        next_edge = best_edge[f"{l+1} </s>"]
        while next_edge != "0 <s>":
            _, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        print(" ".join(tags[::-1]))


if __name__ == "__main__":
    args = sys.argv
    main(args[1])

"""
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN	45
NN --> JJ	27
JJ --> DT	22
NNP --> NN	22
JJ --> NN	12
VBN --> NN	12
NN --> IN	11
NN --> DT	10
NNP --> JJ	8
JJ --> VBN	7
"""