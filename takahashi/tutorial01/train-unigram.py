from typing import Dict, Tuple
from collections import defaultdict
import sys

def count_words(target: str) -> Tuple[Dict[str, int], int]:
    counts = defaultdict(int)
    total_count = 0
    for line in target.split("\n"):
        words = line.strip().split(" ")
        words.append("</s>")
        for word in words:
            counts[word] += 1
            total_count += 1
    return (counts, total_count)

def train_unigram(train_file: str) -> None:
    with open(train_file, "r") as target:
        counts, total_count = count_words(target.read().rstrip("\n"))

    result = []
    for word, count in sorted(counts.items()):
        probability = count / total_count
        result.append(f"{word}\t{probability:.6f}")

    with open("model_file.txt", "w") as f:
        f.write('\n'.join(result) + "\n")

if __name__ == "__main__":
    args = sys.argv
    train_unigram(args[1])
