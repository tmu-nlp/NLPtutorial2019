import math
from collections import defaultdict

model_file = open("./tutorial04/model_file.txt","r").readlines()
input_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/05-test-input.txt","r").readlines()
transition = {}
emission = defaultdict(lambda:0)
possible_tags = {}
lam = 0.95
V = 10 ** 6
for line in model_file:
    type_te, context, word , prob = line.split(" ") 
    possible_tags.setdefault(context,1)
    if type_te == "T":
        transition[context + " " + word] = float(prob)
    else:
        emission[context + " " + word] = float(prob)

for line in input_file:
    words = line.strip().split(" ")
    I = len(words)
    best_score = {}
    best_edge = {}
    best_score["0 <s>"] = 0
    best_edge["0 <s>"]  = None
    for i in range(I):
        for prev_tag in possible_tags.keys():
            for next_tag in possible_tags.keys():
                if str(i) + " " + prev_tag in best_score and prev_tag + " " + next_tag in transition:
                    score = best_score[str(i) + " " + prev_tag] \
                     - math.log2(transition[prev_tag + " " + next_tag]) \
                     - math.log2(lam * emission[next_tag + " " + words[i]]+(1-lam)/V)
                    if f'{i+1} {next_tag}' not in best_score or best_score[f'{i+1} {next_tag}'] > score:
                        best_score[str(i + 1) + " " + next_tag] = score
                        best_edge[str(i + 1) + " " + next_tag] = str(i) + " " + prev_tag
            next_tag = "</s>"
            for tag in possible_tags.keys():
                if  f'{len(words)} {tag}' in best_score and f'{tag} </s>' in transition:
                    P_T = transition[f'{tag} </s>'] 
                    score = best_score[f'{len(words)} {tag}'] - math.log2(P_T)
                    if f'{len(words) + 1} </s>' not in best_score or best_score[f'{len(words) + 1} </s>'] > score:
                        best_score[f'{len(words) + 1} </s>'] = score
                        best_edge[f'{len(words) + 1} </s>'] = f'{len(words)} {tag}'
    
    tags = []
    next_edge = best_edge[f'{len(words)+1} </s>']
    while next_edge != "0 <s>":
        position, tag = next_edge.split(" ")
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    sentence = " ".join(tags)
    print(sentence)
