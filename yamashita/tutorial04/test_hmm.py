import sys
from math import log2
from collections import defaultdict
from itertools import product

transition = defaultdict(float)
emission = defaultdict(float)
possible_tags = defaultdict(int)

m_path = sys.argv[1]
t_path = sys.argv[2]

# モデル読み込み
with open(m_path, 'r', encoding='utf-8') as m_file:
    for line in m_file:
        m_type, context, word, prob = line.split(' ')
        possible_tags[context] = 1
        if m_type == 'T':
            transition[f'{context} {word}'] = float(prob)
        else:
            emission[f'{context} {word}'] = float(prob)

V = 1e6
unk_lambda = 0.05

with open(t_path, 'r', encoding='utf-8') as t_file:
    for line in t_file:
        # 前向きステップ
        words = line.rstrip().split(' ')
        l = len(words)
        best_score = defaultdict(float)
        best_edge = defaultdict(str)
        # BOS
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for i, prev, next_ in product(range(l), possible_tags.keys(), possible_tags.keys()):
            if f'{i} {prev}' not in best_score or f'{prev} {next_}' not in transition:
                continue
            prob_t = transition[f'{prev} {next_}']
            prob_e = (1-unk_lambda) * \
                emission[f'{next_} {words[i]}'] + unk_lambda / V
            score = best_score[f'{i} {prev}'] - log2(prob_t) - log2(prob_e)
            if not best_score[f'{i+1} {next_}'] or best_score[f'{i+1} {next_}'] > score:
                best_score[f'{i+1} {next_}'] = score
                best_edge[f'{i+1} {next_}'] = f'{i} {prev}'

        # EOS
        for key in possible_tags.keys():
            if not transition[f'{key} </s>']:
                continue
            score = best_score[f'{l} {key}'] - log2(transition[f'{key} </s>'])
            if best_score[f'{l+1} </s>'] and best_score[f'{l+1} </s>'] < score:
                continue
            best_score[f'{l+1} </s>'] = score
            best_edge[f'{l+1} </s>'] = f'{l} {key}'

        # 後向きステップ
        tags = []
        next_edge = best_edge[f'{l+1} </s>']
        while next_edge != '0 <s>':
            position, tag = next_edge.split(' ')
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(' '.join(tags))
