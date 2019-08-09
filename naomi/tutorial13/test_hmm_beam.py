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
    best_score = defaultdict(lambda: 0)
    best_edge = defaultdict(lambda: 0)
    possible_tags = defaultdict(lambda: 0)
    for line in open(modelpath, 'r', encoding='utf-8'):
        TE, key1, key2, prob = line.split()
        possible_tags[key1] += 1
        if TE = 'T':
            prob_t[f'{key1} {key2}'] = float(prob)
        else:
            prob_e[f'{key1} {key2}'] = float(prob)
    
    # 前向きステップ
    best_score['0 <s>'] = 0     # <s> で開始
    best_edge['0 <s>'] = NULL
    active_tags[0] = ['<s>']
    
    testpath = '../../test/05-test-input.txt'
    for line in open(testpath, 'r', encoding='utf-8'):
        words = line.rstrip().split()
        
        for i, word in enumerate(words):
            my_best = {}
            for prev in active_tags[i]:
                for next in possible_tags:
                    if f'{i} {prev}' in best_score and f'{prev} {next}' in prob_t:
                        score = best_score[f'{i} {prev}']\
                                - np.log2(prob_t[f'{prev} {next}'])\
                                - np.log2(prob_e[f'{next} {word}'])
                        
                        if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] > score:
                            best_score[f'{i+1} {next}'] = score
                            best_edge[f'{i+1} {next}'] = f'{i} {rev}'
                            my_best[next] = score
            active_tags[i+1] =  best B elements of my_best
