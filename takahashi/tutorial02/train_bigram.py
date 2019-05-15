import sys
from typing import List, Dict, Tuple
from collections import defaultdict

Map = Dict[str, int]
Ngram = List[List[str]]


def get_ngram(words: List[str], n: int) -> Ngram:
    return [words[i : i + n] for i in range(len(words) - n + 1)]


def count_words(train_file: str) -> Tuple[Map, Map]:
    counts = defaultdict(int)  # type: Map
    context_counts = defaultdict(int)  # type: Map
    for line in open(train_file, "r"):
        words = line.rstrip("\n").split(" ")
        for ws in get_ngram(["<s>", *words, "</s>"], 2):
            # bi-gram
            counts[" ".join(ws)] += 1
            context_counts[ws[0]] += 1
            # uni-gram
            counts[ws[1]] += 1
            context_counts[""] += 1
    return counts, context_counts


def train_bigram(train_file: str) -> None:
    counts, context_counts = count_words(train_file)
    result = []  # type: List[str]
    for n_gram, count in sorted(counts.items()):
        context = "" if " " not in n_gram else n_gram.split(" ")[0]
        probability = count / context_counts[context]
        result.append(f"{n_gram}\t{probability:.6f}")

    with open("model_file.txt", "w") as f:
        f.write("\n".join(result) + "\n")


if __name__ == "__main__":
    args = sys.argv
    train_bigram(args[1])
