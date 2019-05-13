# tutorial01 test
import math
from collections import defaultdict
DEBUG = True
UNKNOWN_RATE = 0.05
N = 1000000

# ディクショナリの初期値を0に設定
d = defaultdict(lambda: 0)

# データパス
model_path = "train-input-model.txt" if DEBUG else "wiki-en-train-model.txt"
test_data_path = "../../test/01-test-input.txt" if DEBUG else "../../data/wiki-en-test.word"

# モデルの読み込み
with open(model_path) as f:
    for line in f:
        word, prob = line.split()
        d[word] = prob

# 評価と結果表示
total_word_count = 0
total_unknown_count = 0
h = 0
with open(test_data_path) as f:
    for line in f:
        words = line.split()
        words.append("</s>")
        for word in words:
            total_word_count += 1
            p = UNKNOWN_RATE / N
            if d[word] != 0:
                p += (1 - UNKNOWN_RATE) * float(d[word])
            else:
                total_unknown_count += 1
            h += -math.log(p, 2)

print("entropy = {}".format(h/total_word_count))
print("coverage = {}".format(
    (total_word_count - total_unknown_count)/total_word_count))
