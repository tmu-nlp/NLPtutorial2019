from collections import defaultdict
from math import log

input_path = "../files/test/08-input.txt"
grammar_path = "../files/test/08-grammar.txt"

pre_term = defaultdict(list)
non_term = list()

MINUS_INF = -float("inf")

for rule in open(grammar_path, "r", encoding="utf-8"):
    lhs, rhs, prob = rule.strip().split("\t")
    prob = float(prob)
    rhs_syms = rhs.split()

    if len(rhs_syms) == 1:
        pre_term[rhs].append((lhs, log(prob)))
    else:
        non_term.append((lhs, rhs_syms[0], rhs_syms[1], log(prob)))

for line in open(input_path, "r", encoding="utf-8"):
    words = line.strip().split()
    best_score = defaultdict(lambda: MINUS_INF)
    best_edge = defaultdict(int)

    for i in range(len(words)):
        for (lhs, log_prob) in pre_term[words[i]]:
            best_score[f"{lhs} {i} {i+1}"] = log_prob

    for j in range(2, len(words)+1):
        for i in range(j-1)[::-1]:
            for k in range(i+1, j):
                for sym, lsym, rsym, log_prob in non_term:
                    lsym_ik = f"{lsym} {i} {k}"
                    rsym_kj = f"{rsym} {k} {j}"
                    if best_score[lsym_ik] > MINUS_INF and best_score[rsym_kj] > MINUS_INF:
                        my_lp = best_score[lsym_ik] + best_score[rsym_kj] + log_prob

                        if my_lp > best_score[rsym]:
                            sym_ij = f"{sym} {i} {j}"
                            best_score[sym_ij] = my_lp
                            best_edge[sym_ij] = (lsym_ik, rsym_kj)

    def print_sexp(sym_ij):
        sym, index, _ = sym_ij.split()
        i = int(index)
        if sym_ij in best_edge:
            return f"({sym} {print_sexp(best_edge[sym_ij][0])} {print_sexp(best_edge[sym_ij][1])})"
        else:
            return f"({sym} {words[i]})"

    print(print_sexp(f"S 0 {len(words)}"))