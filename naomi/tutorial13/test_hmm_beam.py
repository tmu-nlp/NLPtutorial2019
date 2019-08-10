from collections import defaultdict
import numpy as np

# 入力
# train-input.txt
# a_X b_Y a_Z
# train-answer.txt
# T <s> X 1.000000
# E X a 0.666667
def train_hmm():
    in_path = '../../test/05-train-input.txt'
    in_path = '../../data/wiki-en-train.norm_pos'
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


def test_hmm_beam():
    modelpath = 'trained_model.txt'
    prob_e = defaultdict(lambda: 0)
    prob_t = defaultdict(lambda: 0)
    possible_tags = defaultdict(lambda: 0)
    for line in open(modelpath, 'r', encoding='utf-8'):
        TE, key1, key2, prob = line.split()
        possible_tags[key1] += 1
        if TE == 'T':
            prob_t[f'{key1} {key2}'] = float(prob)
        else:
            prob_e[f'{key1} {key2}'] = float(prob)
    



    l1 = 0.9
    # 未知語を含んだ語彙数
    V = 1e6

    tags_list = []
    
    testpath = '../../test/05-test-input.txt'
    testpath = '../../data/wiki-en-test.norm'
    # 前向きステップ
    for line in open(testpath, 'r', encoding='utf-8'):
        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)
        active_tags = defaultdict(lambda: 0)

    
        best_score['0 <s>'] = 0     # <s> で開始
        best_edge['0 <s>'] = None
        active_tags[0] = ['<s>']

        words = line.rstrip().split()
        
        for i, word in enumerate(words):
            my_best = {}
            for prev in active_tags[i]:
                for nxt in possible_tags:
                    if f'{i} {prev}' not in best_score or\
                       f'{prev} {nxt}' not in prob_t:
                        continue

                    # 遷移確率
                    Pt = prob_t[f'{prev} {nxt}']

                    # 未知語を含むときの生成確率
                    Pe = l1 * prob_e[f'{nxt} {word}'] + (1-l1)/V
                    score = (best_score[f'{i} {prev}']
                             - np.log2(Pt)
                             - np.log2(Pe))
                    
                    if f'{i+1} {nxt}' in best_score and\
                       best_score[f'{i+1} {nxt}'] <= score:
                        continue
                    best_score[f'{i+1} {nxt}'] = score
                    best_edge[f'{i+1} {nxt}'] = f'{i} {prev}'
                    my_best[nxt] = score
            
            sorted_tags = [k for k in sorted(my_best, key=my_best.get, reverse=False)]
            active_tags[i+1] = sorted_tags[:3]

        # 文末記号への遷移を考える
        for tag in possible_tags:
            if '{0} {1}'.format((i+1), tag) not in best_score \
               or tag + ' </s>' not in prob_t:
                continue

            # 遷移確率
            Pt = prob_t[tag + ' </s>']

            # 未知語を含むときの生成確率
            Pe = l1 * prob_e[tag + ' </s>'] + (1-l1)/V

            # スコアの計算
            score = (best_score['{0} {1}'.format(i+1, tag)]
                     - np.log2(Pt)
                     - np.log2(Pe))
            
            # ベストスコアのチェック（小さいほどよい）
            if f'{i+1+1} </s>' in best_score \
               and best_score[f'{i+1+1} </s>'] <= score:
                continue

            # ベストスコアの更新
            best_score[str(i+1+1)+' </s>'] = score
            best_edge[str(i+1+1)+' </s>'] = '{0} {1}'.format(i+1, tag)

        # 後ろ向きステップ
        tags = []

        next_edge = best_edge[str(i+1+1)+' </s>']

        while next_edge != '0 <s>':
            # このエッジの品詞を出力に追加
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]

        # 順番を入れ替える
        tags = tags[::-1]
        tags_list.append(' '.join(tags))

    return tags_list


if __name__ == "__main__":
    train_hmm()
    tags_list = test_hmm_beam()
    with open('tutorial13.txt', 'w+', encoding='utf-8') as fout:
        for tags in tags_list:
            print(tags, file=fout)

# Accuracy: 90.51% (4130/4563)

# Most common mistakes:
# NNS --> NN      55
# NN --> JJ       29
# NNP --> NN      25
# JJ --> DT       24
# JJ --> NN       15
# VBN --> NN      12
# JJ --> VBN      11
# NN --> IN       10
# NN --> DT       10
# VBG --> NN      9