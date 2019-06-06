# test-hmm.py
import math
from collections import defaultdict

input_model_path = "tutorial04.txt"
# test_input_path = "../../test/05-test-input.txt"
test_input_path = "../../data/wiki-en-test.norm"
output_path = "my_answer.pos"
UNKNOWN_RATE = 0.05
N = 1e6

transition = defaultdict(lambda: 0)
emission = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)

with open(input_model_path) as fr:
    for line in fr:
        typ, context, word, prob = line.split()
        possible_tags[context] = 1
        if typ == "T":
            transition[f"{context} {word}"] = float(prob)
        else:
            emission[f"{context} {word}"] = float(prob)

with open(test_input_path) as f, open(output_path, mode="w") as fw:
    for line in f:
        words = line.split()
        l = len(words)

        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)

        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None

        for i in range(l):
            for prev_tag in possible_tags.keys():
                for next_tag in possible_tags.keys():
                    if f"{i} {prev_tag}" not in best_score or f"{prev_tag} {next_tag}" not in transition:
                        continue
                    pt = transition[f"{prev_tag} {next_tag}"]
                    pe = (1 - UNKNOWN_RATE) * emission[f"{next_tag} {words[i]}"] + UNKNOWN_RATE / N
                    
                    score = best_score[f"{i} {prev_tag}"] - math.log(pt, 2) - math.log(pe, 2)
                    if f"{i+1} {next_tag}" not in best_score or best_score[f"{i+1} {next_tag}"] > score:
                        best_score[f"{i+1} {next_tag}"] = score
                        best_edge[f"{i+1} {next_tag}"] = f"{i} {prev_tag}"
        for tag in possible_tags.keys():
            if f"{l} {tag}" in best_score and f"{tag} </s>" in transition:
                pt = transition[f"{tag} </s>"]
                pe = (1 - UNKNOWN_RATE) * emission[f"{tag} </s>"] + UNKNOWN_RATE / N
                score = best_score[f"{l} {tag}"] - math.log(pt, 2) - math.log(pe, 2)
                if f"{l+1} </s>" not in best_score or best_score[f"{l+1} </s>"] > score:
                    best_score[f"{l+1} </s>"] = score
                    best_edge[f"{l+1} </s>"] = f"{l} {tag}"
        tags = []
        next_edge = best_edge[f"{l+1} </s>"]
        while next_edge != "0 <s>" and next_edge != None:
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags = tags[::-1]
        fw.write(" ".join(tags) + "\n")