# tutorial00
# ファイルの中の単語の頻度を数えるプログラムを作成
from collections import defaultdict
DEBUG = True

# ディクショナリの初期値を0に設定
d = defaultdict(lambda: 0)

# デバッグモードの場合は短文を入力にする
input_path = "../test/00-input.txt" \
    if DEBUG else "../data/wiki-en-train.word"

with open(input_path) as f:
    # 1行ずつ
    for s_line in f:
        # 文を単語に分割
        for word in s_line.split():
            # ディクショナリに1を足していく
            d[word] += 1

# 結果を表示
for (key, value) in d.items():
    print(key, value)