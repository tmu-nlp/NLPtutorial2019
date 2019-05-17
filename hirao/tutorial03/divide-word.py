# divide-word.py
from collections import defaultdict
import math

UNKNOWN_RATE = 0.05
N = 1000000

word_data_path = "../../data/wiki-ja-train.word"
save_model_path = "wiki-ja-train-model.txt"
input_path = "../../data/wiki-ja-test.txt"
output_path = "tutorial03.txt"
# ディクショナリの初期値を0に設定
d = defaultdict(lambda: 0)
total_count = 0

# テキストの読み込み
with open(word_data_path) as f:
    for line in f:
        words = line.split()
        words.append("</s>")
        # 特定の単語の出現数と全体の単語数をカウント
        for word in words:
            d[word] += 1
            total_count += 1

# 単語と出現率を出力
with open(save_model_path, mode='w') as f:
    for key, value in d.items():
        prob = value / total_count
        f.write("{} {}\n".format(key, prob))


uni_probs = defaultdict(lambda: 0)
# unigramのモデルを読み込み
with open(save_model_path) as f:
    for line in f:
        unigram, prob = line.split()
        uni_probs[unigram] = float(prob)

with open(input_path) as f, open(output_path, mode="w") as fw:
    # 前向きステップ
    for line in f:
        best_edge = [None] * len(line)
        best_score = [0] * len(line)
        for word_end in range(1, len(line)):
            best_score[word_end] = 1e10
            for word_begin in range(word_end):
                word = line[word_begin : word_end]
                if word in uni_probs.keys() or len(word) == 1:
                    prob = (1 - UNKNOWN_RATE) * uni_probs[word] + (UNKNOWN_RATE / N)
                    my_score = best_score[word_begin] - math.log(prob, 2)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        '''
        ex.
        best_score = [0, 5.770780162668462, 11.541560325336924, 8.80043975151574]
        best_edge = [None, (0, 1), (1, 2), (1, 3)]
        '''
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge:
            # このエッジの部分文字列を追加
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words = words[::-1]
        out_put_str = " ".join(words)
        fw.write(out_put_str + "\n")

'''
$ ../../script/gradews.pl ../../data/wiki-ja-test.word tutorial03.txt
Sent Accuracy: 23.81% (20/84)
Word Prec: 71.88% (1943/2703)
Word Rec: 84.22% (1943/2307)
F-meas: 77.56%
Bound Accuracy: 86.30% (2784/3226)
'''