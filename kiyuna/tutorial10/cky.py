import os
import re
import sys
import math
from nltk.tree import Tree
from itertools import islice
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .
INF = float('inf')


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stdout.write("\33[92m" + text + "\33[0m")


def print_ST(sym_ij, best_edge, words):
    """ #10 p68 """
    sym, i, _ = re.sub("[()]", "", sym_ij).split()
    if sym_ij in best_edge:
        left = print_ST(best_edge[sym_ij][0],best_edge, words)
        right = print_ST(best_edge[sym_ij][1], best_edge, words)
        return f'({sym} {left} {right})'
    else:
        return f'({sym} {words[int(i)]})'


def load_grammer(grammar_file):
    """ #10 p65 """
    nonterm = []
    preterm = defaultdict(list)     # preterm[右] := [(左, 確率) ...]
    for rule in open(grammar_file):
        lhs, rhs, prob = rule.split('\t')   # P(左 -> 右) = 確率
        rhs_symbols = rhs.split()
        prob = float(prob)
        if len(rhs_symbols) == 1:           # 前終端記号
            preterm[rhs] += [(lhs, math.log(prob))]
        else:                               # 非終端記号
            nonterm += [(lhs, rhs_symbols[0], rhs_symbols[1], math.log(prob))]
    return nonterm, preterm


def cky(grammar_file, input_file, s=0, t=57):
    """ #10 p66-67 """
    nonterm, preterm = load_grammer(grammar_file)
    for line in islice(open(input_file), s, t):
        # 前終端記号を追加
        words = line.split()
        # best_score[sym_{i, j}] := 最大対数確率
        best_score = defaultdict(lambda: -INF)
        # best_edge[sym_{i, j}] := (lsym_{i, k}, rsym_{k, j})
        best_edge = {}
        for i in range(len(words)):
            if preterm[words[i]]:
                for lhs, log_prob in preterm[words[i]]:
                    best_score[f'{lhs} ({i} {i+1})'] = log_prob
        # 非終端記号の組み合わせ
        for j in range(2, len(words) + 1):
            for i in range(j - 2, -1, -1):
                for k in range(i + 1, j):
                    # log(P(sym -> lsym rsym)) = log prob
                    for sym, lsym, rsym, logprob in nonterm:
                        par = f'{sym} ({i} {j})'
                        left = f'{lsym} ({i} {k})'
                        right = f'{rsym} ({k} {j})'
                        # 両方の子供の確率が 0 より大きい
                        if best_score[left] == -INF:
                            continue
                        if best_score[right] == -INF:
                            continue
                        # このノード・辺の対数確率を計算
                        my_lp = best_score[left] + best_score[right] + logprob
                        # この辺が確率最大のものなら更新
                        if my_lp > best_score[par]:
                            best_score[par] = my_lp
                            best_edge[par] = (left, right)
        res = print_ST(f'S (0 {len(words)})', best_edge, words)
        print(res)
        # t = Tree.fromstring(res)
        # print(t)
        # t.draw()


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        message('[*] test')
        grammar_file = "../../test/08-grammar.txt"
        input_file = "../../test/08-input.txt"
    else:
        message('[*] main')
        grammar_file = "../../data/wiki-en-test.grammar"
        input_file = "../../data/wiki-en-short.tok"

    cky(grammar_file, input_file, 0, 2)


'''RESULT（一部）
(S (PP (IN Among) (NP (DT these) (NP' (, ,) (NP' (JJ supervised) (NP' (NN lea
rning) (NNS approaches)))))) (S' (VP (VBP have) (VP (VBN been) (VP' (NP (DT t
he) (NP' (ADJP (RBS most) (JJ successful)) (NNS algorithms))) (PP (TO to) (NP
_NN date))))) (. .)))
(S (NP (JJ Current) (NN accuracy)) (S' (VP (VBZ is) (ADJP (JJ difficult) (S_V
P (TO to) (VP (VB state) (PP (IN without) (NP (NP (DT a) (NN host)) (PP (IN o
f) (NP_NNS caveats)))))))) (. .)))
'''
