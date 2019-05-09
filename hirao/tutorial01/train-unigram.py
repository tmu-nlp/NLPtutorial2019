# tutorial01 train
from collections import defaultdict
DEBUG = True

# ディクショナリの初期値を0に設定
d = defaultdict(lambda: 0)

# データパス
train_data_path = "../../test/01-train-input.txt" if DEBUG else "../../data/wiki-en-train.word"
save_model_path = "train-input-model.txt" if DEBUG else "wiki-en-train-model.txt"

total_count = 0

# テキストの読み込み
with open(train_data_path) as f:
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
print("Saved model in {}".format(save_model_path))
