from collections import defaultdict
from math import log

def print_sym(sym_ij):
    sym, index, _ = sym_ij.split()
    i = int(index)

    if sym_ij in best_edge:
        return f'{sym} {print_sym(best_edge[sym_ij][0])} {print_sym(best_edge[sym_ij][1])}'
    else:
        return f'{sym} {words[index]}'


# (左, 右１, 右２, 確率)
nonterm = []
# pre[右] = [(左, 確率)...]
preterm = defaultdict(list)

for line in open('../../test/08-grammar.txt', 'r', encoding='utf-8'):
    lhs, rhs_, prob = line.strip().split('\t')
    prob = float(prob)
    rhs = rhs_.split()
    if len(rhs) == 1:
        preterm[rhs[0]].append((lhs, prob))
    else:
        nonterm.append((lhs, rhs[0], rhs[1], prob))

for line in open('../../test/08-input.txt', 'r', encoding='utf-8'):
    words = line.strip().split()

    # {sym_ij: 最大対数確率}
    best_score = {}
    # {sym_ij: (lsym_ik, rsym_kj)}
    best_edge = {}

    # 前終端記号を追加
    for i, word in enumerate(words):
        for (lhs, prob) in preterm[word]:
            if prob > 0:
                best_score[f'{lhs} {i} {i+1}'] = log(prob)
 
    # jはスパンの右側
    for j in range(2, len(words)+1):
        # jはスパンの左側
        for i in range(j-1)[::-1]:
            for k in range(i+1, j):
                for sym, lsym, rsym, prob in nonterm:
                    lsym_ik = f'{lsym} {i} {k}'
                    rsym_kj = f'{rsym} {k} {j}'
                    if best_score[lsym_ik] > -float('inf') and best_score[rsym_kj] > -float('inf'):
                        my_lp = best_score[lsym_ik] + best_score[rsym_kj] + log(prob)

                        if my_lp > best_score[rsym]:
                            sym_ij = f'{sym} {i} {j}'
                            best_score[sym_ij] = my_lp
                            best_edge[sym_ij] = (lsym_ik, rsym_kj)

    print(print_sym(f"S 0 {len(words)}"))

    print('')
