# 1 演習問題
from collections import defaultdict
DEBUG = True

d = defaultdict(lambda: 0)


input_path = "../test/00-input.txt" \
    if DEBUG else "../data/wiki-en-train.word"

with open(input_path) as f:
    for s_line in f:
        for c in s_line.split():
            d[c] += 1

for (key, value) in d.items():
    print(key, value)