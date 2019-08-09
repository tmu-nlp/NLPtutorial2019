from collections import defaultdict

# 入力
# train-input.txt
# a_X b_Y a_Z
# train-answer.txt
# T <s> X 1.000000
# E X a 0.666667

in_path = '../../test/05-train-input.txt'
out_path = 'trained_model.txt'

emission = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)

for line in open(in_path, 'r', encoding='utf-8'):
    word_tag_list = line.rstrip().split()

    # 文頭記号
    previous = '<s>'
    # 単語_タグずつ
    for word_tag in word_tag_list:
        # 出力（今の単語 → 今のtag）
        emission[word_tag] += 1
        word, tag = word_tag.split('_')
        # 遷移（前のtag →　今のtag）
        transition[f'{previous} {tag}'] += 1
        # 前のtag
        context[previous] += 1
        # 次のステップのために保存
        previous = tag
    # 文末記号
    context[previous] += 1
    transition[f'{tag} </s>'] += 1

with open(out_path, 'w+', encoding='utf-8') as f:

    # 遷移（前の品詞から今の品詞）確率の計算
    for (key, value) in transition.items():
        [previous, current] = key.split()
        print('T {0} {1:.5f}'.format(key, value/context[previous]), file=f)
    # 生成（その品詞からその単語）確率の計算
    for (key, value) in emission.items():
        [word, tag] = key.split('_')
        print('E {0} {1} {2:.5f}'.format(tag, word, value/context[tag]), file=f)

# # 前向きステップ
# best_score[“0 <s>”] = 0 # <s> で開始
# best_edge[“0 <s>”] = NULL
# active_tags[0] = [ “<s>” ]
# for i in 0 … I-1:
# make map my_best
# for each prev in keys of active_tags[i]
# for each next in keys of possible_tags
# if best_score[“i prev”] and transition[“prev next”] exist
# score = best_score[“i prev”] +
# -log P
# T
# (next|prev) + -log P
# E
# (word[i]|next)
# if best_score[“i+1 next”] is new or > score
# best_score[“i+1 next”] = score
# best_edge[“i+1 next”] = “i prev”
# my_best[next] = score
# active_tags[i+1] = best B elements of my_best
# # </s> で同等の処理を行う