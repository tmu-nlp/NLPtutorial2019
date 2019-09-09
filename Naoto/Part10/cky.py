from math import log
from collections import defaultdict
import sys
from nltk.tree import Tree
from itertools import islice

INF = float('inf')


def message(text='', CR=False):
    text = '\r' + text if CR else text + '\n'
    sys.stdout.write('\33[92m' + text + '\33[0m')


class Parsing:
    def __init__(self, grammar_path, input_path):
        self.grammar_path = grammar_path
        self.input_path = input_path

    def read(self):
        """
        "lhs \t rhs \t prob \n" 形式の文法を読み込む
        #10 p.65
        """
        self.nonterm = []  # (左, 右1, 右2, 確率)の非終端記号
        self.preterm = defaultdict(list)  # pre[ 右 ] = [ ( 左 , 確率 ) ...] 形式のマップ
        for rule in map(lambda x: x.rstrip(), open(self.grammar_path)):
            lhs, rhs, prob = rule.split('\t')  # P(左 → 右)= 確率
            prob = float(prob)
            rhs = rhs.split(' ')
            if len(rhs) == 1:  # 前終端記号
                rhs = str(rhs[0])
                self.preterm[rhs] += [(lhs, log(prob))]
            else:  # 非終端記号
                self.nonterm += [(lhs, rhs[0], rhs[1], log(prob))]

    def cky(self, s=0, t=57):
        """
        非終端記号の組み合わせ
        #10 p.67
        """
        for words in islice(map(lambda x: x.rstrip().split(), open(self.input_path)), s, t):
            len_words = len(words)
            best_score = defaultdict(lambda: -INF)  # 引数 =sym(i,j) 値 = 最大対数確率
            self.best_edge = {}  # 引数 =sym(i,j) 値 =(lsym(i,k), rsym(k,j))
            # 前終端記号を追加
            len_words = len(words)
            for i in range(len_words):
                if self.preterm[words[i]]:
                    for lhs, log_prob in self.preterm[words[i]]:
                        best_score[f'{lhs} {i} {i+1}'] = log_prob
            for j in range(2, len_words + 1):  # j はスパンの右側
                for i in range(j-2, -1, -1):  # i はスパンの左側 (右から左へ処理!)
                    for k in range(i + 1, j):  # k は rsym の開始点
                        # 各文法ルールを展開 :log(P(sym → lsym rsym)) = logprob
                        for sym, lsym, rsym, logprob in self.nonterm:
                            par = f'{sym} {i} {j}'
                            left = f'{lsym} {i} {k}'
                            right = f'{rsym} {k} {j}'
                            # 両方の子供の確率が 0 より大きい
                            if best_score[left] == -INF:
                                continue
                            if best_score[right] == -INF:
                                continue
                            # このノード・辺の対数確率を計算
                            # print('eval')
                            my_lp = best_score[left] + best_score[right] + logprob
                            if my_lp > best_score[par]:
                                best_score[par] = my_lp
                                self.best_edge[par] = (left, right)
            # print(f'self.best_edge {self.best_edge}')
            # print(f'len(words) {len(words)}')
            res = self.print_tree(f'S 0 {len(words)}', words)
            print(res)
            t = Tree.fromstring(res)
            print(t)
            t.draw()

    def print_tree(self, sym_i_j, words):
        sym, i, _ = sym_i_j.split(' ')
        i = int(i)
        if sym_i_j in self.best_edge:  # 非終端記号
            return '(' + sym + ' ' + \
                   self.print_tree(self.best_edge[sym_i_j][0], words) + ' ' + \
                   self.print_tree(self.best_edge[sym_i_j][1], words) + ')'
        else:  # 終端記号
            return '(' + sym + ' ' + words[i] + ')'


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        message('[*] test')
        grammar_path = '../../../nlptutorial/test/08-grammar.txt'
        input_path = '../../../nlptutorial/test/08-input.txt'
    else:
        message('[*] main')
        grammar_path = '../../../nlptutorial/data/wiki-en-test.grammar'
        input_path = '../../../nlptutorial/data/wiki-en-short.tok'

    parsing = Parsing(grammar_path, input_path)
    parsing.read()
    parsing.cky(1, 3)
