from collections import defaultdict
import math
from tqdm import tqdm


class cky:
    def __init__(self, grammar_file, input_file):
        self.grammar_file = grammar_file
        self.input_file = input_file
        self.nonterm = []
        self.preterm = defaultdict(list)
        self.best_score = defaultdict(lambda: 0)
        self.best_edge = defaultdict(lambda: 0)

    def load_grammar(self):
        for line in open(self.grammar_file):
            lhs, rhs, prob = line.strip().split('\t')
            rhs = rhs.split(' ')

            if len(rhs) == 1:
                self.preterm[rhs[0]].append((lhs, math.log2(float(prob))))
            else:
                self.nonterm.append((lhs, rhs[0], rhs[1], math.log2(float(prob))))

    def line_loop(self):
        lines = [line.strip().split(' ') for line in open(self.input_file)]
        with open('out.txt', 'w') as f:
            for words in tqdm(lines, desc='line_loop'):
                print(self.predict_tree(words), file=f)

    def predict_tree(self, words):
        for i, word in enumerate(words):
            for lhs, prob in self.preterm[word]:
                self.best_score[f'{i}|{i+1}|{lhs}'] = prob

        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for sym, lsym, rsym, logprob in self.nonterm:
                        key = f'{i}|{j}|{sym}'
                        l_key = f'{i}|{k}|{lsym}'
                        r_key = f'{k}|{j}|{rsym}'
                        if (l_key in self.best_score) and (r_key in self.best_score):
                            my_lp = self.best_score[l_key] + self.best_score[r_key] + logprob
                            if (key not in self.best_score) or (my_lp > self.best_score[key]):
                                self.best_score[key] = my_lp
                                self.best_edge[key] = (l_key, r_key)

        return self.create_tree(f'0|{len(words)}|S', words)

    def create_tree(self, key, words):
        sym = key.split('|')[2]
        print(sym)
        if key in self.best_edge:
            lkey, rkey = self.best_edge[key]
            lstruct = self.create_tree(lkey, words)
            rstruct = self.create_tree(rkey, words)
            return f'({sym} {lstruct} {rstruct})'
        else:
            i = int(key.split('|')[0])
            return f'({sym} {words[i]})'


if __name__ == '__main__':
    cky = cky('../../test/08-grammar.txt', '../../test/08-input.txt')
    # cky = cky('../../data/wiki-en-test.grammar', '../../data/wiki-en-short.tok')
    cky.load_grammar()
    cky.line_loop()
    print(cky.best_edge)
