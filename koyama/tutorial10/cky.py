from collections import defaultdict
import math

#grammer_path = '../../test/08-grammar.txt'
grammer_path = '../../data/wiki-en-test.grammar'

#input_path = '../../test/08-input.txt'
input_path = '../../data/wiki-en-short.tok'

def PRINT(sym_i_j):
    sym, i, j = sym_i_j.split()
    if sym_i_j in best_edge:
        return f'({sym} {PRINT(best_edge[sym_i_j][0])} {PRINT(best_edge[sym_i_j][1])})'
    else:
        return f'({sym} {words[int(i)]})'

nonterm = []
preterm = defaultdict(list)
with open(grammer_path, 'r') as grammar_file:
    for rule in grammar_file:
        lhs, rhs, prob = rule.rstrip().split('\t')
        rhs_symbols = rhs.split()
        if len(rhs_symbols) == 1:
            preterm[rhs].append((lhs, math.log2(float(prob))))
        else:
            nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log2(float(prob))))

with open(input_path, 'r') as input_file, open('answer-file', 'w') as ans_file:
    for line in input_file:
        words = line.rstrip().split()
        best_score = defaultdict(lambda: -math.inf)
        best_edge = {}
        for i in range(len(words)):
            for lhs, log_prob in preterm[words[i]]:
                best_score[f'{lhs} {i} {i+1}'] = log_prob
        for j in range(2, len(words)+1):
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    for sym, lsym, rsym, logprob in nonterm:
                        if best_score[f'{lsym} {i} {k}'] > -math.inf and best_score[f'{rsym} {k} {j}'] > -math.inf:
                            my_lp = best_score[f'{lsym} {i} {k}'] + best_score[f'{rsym} {k} {j}'] + logprob
                            if my_lp > best_score[f'{sym} {i} {j}']:
                                best_score[f'{sym} {i} {j}'] = my_lp
                                best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')
        print(PRINT(f'S 0 {len(words)}'), file=ans_file)
