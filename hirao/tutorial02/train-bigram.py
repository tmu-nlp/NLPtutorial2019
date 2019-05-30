# train-bigram.py
from collections import defaultdict

DEBUG = False
# データパス
train_data_path = "../../test/02-train-input.txt" if DEBUG else "../../data/wiki-en-train.word"
save_model_path = "train-input-model.txt" if DEBUG else "wiki-en-train-model.txt"

counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)

# テキストの読み込み
with open(train_data_path) as f:
    for line in f:
        words = line.strip().split()
        # 終端記号を追加
        words.append("</s>")
        # 文頭記号を追加
        words.insert(0, "<s>")
        for i in range(len(words) - 1):
            counts["{} {}".format(words[i], words[i+1])] += 1
            context_counts["{}".format(words[i])] += 1
            counts["{}".format(words[i+1])] += 1
            context_counts[""] += 1

ans_list = []
with open(save_model_path, mode="w") as f:
    for ngram, count in counts.items():
        context = ngram.split()
        if len(context) > 1:
            context = context[0]
        else:
            context = ""
        probability = counts[ngram] / context_counts[context]
        ans_list.append([ngram, probability])

    prob_list.sort(key=lambda x: x[0])
    for ngram, probability in prob_list:
        output = "{}\t{}\n".format(ngram, probability)
        f.write(output)
