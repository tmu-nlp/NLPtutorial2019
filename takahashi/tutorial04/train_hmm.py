from collections import defaultdict
from typing import Dict
import sys


def train(filename: str) -> None:
    context = defaultdict(int)  # type: Dict[str, int]
    transition = defaultdict(int)  # type: Dict[str, int]
    emit = defaultdict(int)  # type: Dict[str, int]

    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        prev = "<s>"
        context[prev] += 1

        for wordtag in line.split():
            word, tag = wordtag.split("_")
            transition[prev + " " + tag] += 1
            context[tag] += 1
            emit[tag + " " + word] += 1
            prev = tag
        transition[prev + " </s>"] += 1

    with open("model.txt", "w", encoding="utf-8") as f:
        # 遷移確率を出力
        for key, value in transition.items():
            prev, word = key.split()
            f.write(f"T\t{key}\t{value / context[prev]:.6}" + "\n")

        # 生成確率を出力
        for key, value in emit.items():
            prev, word = key.split()
            f.write(f"E\t{key}\t{value / context[prev]:.6}" + "\n")


if __name__ == "__main__":
    args = sys.argv
    train(args[1])
