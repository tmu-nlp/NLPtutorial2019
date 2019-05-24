import sys
import math
from collections import defaultdict

transition = defaultdict(float)
emission = defaultdict(float)
possible_tags = defaultdict(int)

m_path = sys.argv[1]
t_path = sys.argv[2]

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
        words = line.rstrip().split(' ')
        l = len(words)
        best_score = defaultdict(float)
        best_edge = defaultdict(str)
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for i in range(l):
            for prev in possible_tags.keys():
                for next_ in possible_tags.keys():
                    if f'{i} {prev}' not in best_score or f'{prev} {next_}' not in transition:
                        continue
                    prob_t = transition[f'{prev} {next_}']
                    prob_e = (1-unk_lambda) * \
                        emission[f'{next_} {words[i]}'] + unk_lambda / V
                    score = best_score[f'{i} {prev}'] + - \
                        math.log2(prob_t) + -math.log2(prob_e)
                    if not best_score[f'{i+1} {next_}'] or best_score[f'{i+1} {next_}'] > score:
                        best_score[f'{i+1} {next_}'] = score
                        best_edge[f'{i+1} {next_}'] = f'{i} {prev}'
                        # continue
                    # best_score[f'{i+1} {next_}'] = score
                    # best_edge[f'{i+1} {next_}'] = f'{i} {prev}'

        for key in possible_tags.keys():
            if not transition[f'{key} </s>']:
                continue
            score = best_score[f'{l} {key}'] + - \
                math.log2(transition[f'{key} </s>'])
            if best_score[f'{l+1} </s>'] and best_score[f'{l+1} </s>'] < score:
                continue
            best_score[f'{l+1} </s>'] = score
            best_edge[f'{l+1} </s>'] = f'{l} {key}'

        tags = []
        next_edge = best_edge[f'{l+1} </s>']
        while next_edge != '0 <s>':
            position, tag = next_edge.split(' ')
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(' '.join(tags))
