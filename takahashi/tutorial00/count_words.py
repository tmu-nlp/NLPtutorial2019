from typing import Dict
import sys

def word_count(filename: str) -> Dict[str, int]:
    counts = {}

    with open(filename, "r") as target:
        for line in target.readlines():
            words = line.strip().split(" ")

            for w in words:
                if w in counts.keys():
                    counts[w] += 1
                else:
                    counts[w] = 1

    return counts

if __name__ == "__main__":
    args = sys.argv
    for key, value in sorted(word_count(args[1]).items()):
        print(f"{key}\t{value}")



