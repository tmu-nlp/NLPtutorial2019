from collections import defaultdict
from operator import itemgetter
from math import log


def load_test():
    test_file = "../files/data/wiki-en-test.norm"
    for line in open(test_file, "r", encoding="utf-8"):
        words = line.strip().split()
        yield words


def beam_search(words, possible_tags, trans, emiss):
    unk = 0.05
    vocab = 1e6

    l = len(words)
    best_score, best_edge = {"0 <s>": 0}, {"0 <s>": None}
    active_tags = [["<s>"]]

    for i in range(l):
        my_best = dict()
        for prv in active_tags[i]:
            for nxt in possible_tags:
                if f"{i} {prv}" not in best_score or f"{prv} {nxt}" not in trans:
                    continue

                t_prob = trans[f"{prv} {nxt}"]
                e_prob = (1 - unk) * emiss[f"{nxt} {words[i]}"] + unk / vocab
                score = best_score[f"{i} {prv}"] - log(t_prob) - log(e_prob)
                if f"{i+1} {nxt}" not in best_score or best_score[f"{i+1} {nxt}"] > score:
                    best_score[f"{i+1} {nxt}"], best_edge[f"{i+1} {nxt}"] = score, f"{i} {prv}"
                    my_best[nxt] = score
        best = sorted(my_best.items(), key=itemgetter(1))[:BEAM]
        active_tags.append([k for k,_ in best])

    for tag in active_tags[-1]:
        if f"{l} {tag}" in best_score and f"{tag} </s>" in trans:
            score = best_score[f"{l} {tag}"] - log(trans[f"{tag} </s>"])
            if f"{l+1} </s>" not in best_score or best_score[f"{l+1} </s>"] > score:
                best_score[f"{l+1} </s>"], best_edge[f"{l+1} </s>"] = score, f"{i} {prv}"

    tags, next_edge = list(), best_edge[f"{l+1} </s>"]
    while next_edge != "0 <s>":
        _, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags = tags[::-1]

    return tags


def main():
    trans, emiss = defaultdict(float), defaultdict(float)
    possible_tags = set()

    for line in open(model, "r", encoding="utf-8"):
        type_, context_word, prob = line.strip().split("\t")
        context, word = context_word.split()
        possible_tags.add(context)
        if type_ == "T":
            trans[f"{context} {word}"] = float(prob)
        else:
            emiss[f"{context} {word}"] = float(prob)

    with open("result.txt", "w", encoding="utf-8") as out_file:
        for words in load_test():
            y_hat = beam_search(words, possible_tags, trans, emiss)
            print(" ".join(y_hat) + " .", file=out_file)


if __name__ == "__main__":
    model = "model.txt"
    BEAM = 5
    main()
