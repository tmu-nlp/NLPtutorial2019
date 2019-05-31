from collections import defaultdict
import math


def model_read(model):
    with open(model, "r") as f:
        transition = defaultdict(lambda: 0)
        emission = defaultdict(lambda: 0)
        possible_tags = defaultdict(lambda: 0)
        for line in f:
            type_, context, word, prob = line.rstrip().split(" ")
            possible_tags[context] = 1
            if type_ == "T":
                transition[f"{context} {word}"] = prob
            else:
                emission[f"{context} {word}"] = prob
    return transition, emission


def foward_back_step(transition: {}, emission: {}, lines: [], path: str):
    with open(path, "w") as f:
        count = 0
        λ = 0.95
        λ_unk = 1 - λ
        N = 10000
        p_ns = list(emission.items())
        for i in range(len(p_ns)):
            p_ns[i] = list(p_ns[i])
            p_ns[i][0:1] = p_ns[i][0].split(" ")
        p_ns.sort(key=lambda x: x[1])
        emission_word_key = defaultdict(lambda: λ_unk)
        for p_n in p_ns:
            if p_n[1] in emission_word_key:
                emission_word_key[p_n[1]].extend([p_n[0], λ*p_n[2]])
            else:
                emission_word_key[p_n[1]] = [p_n[0], λ*p_n[2]]
        emission_word_key["<s>"] = ["<s>"]
        for line in lines:
            words = line.rstrip().split(" ")
            words.insert(0, "<s>")
            words.append("</s>")
            l = len(words)
            best_score = {}
            best_edge = {}
            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = None
            for i in range(l-2):
                for p in emission_word_key[words[i]][::2]:
                    for n in emission_word_key[words[i+1]][::2]:
                        # print(words[i], p, words[i+1], n)
                        if not(f"{i} {p}" in best_score and f"{p} {n}" in transition):
                            continue
                        # print("T", p, n)
                        # if n == "</s>":
                        #     print(n, words[i+1])
                        #     print(emission[f"{n} {words[i+1]}"])
                        score = best_score[f"{i} {p}"] + -math.log(float(transition[f"{p} {n}"])) + -math.log(float(emission[f"{n} {words[i+1]}"]))
                        if not(f"{i+1} {n}" in best_score) or best_score[f"{i+1} {n}"] < score:
                            # print("S", p, n)
                            best_score[f"{i+1} {n}"] = score
                            best_edge[f"{i+1} {n}"] = f"{i} {p}"
            i = l-2
            for p in emission_word_key[words[i]][::2]:
                n = "</s>"
                if not(f"{i} {p}" in best_score and f"{p} {n}" in transition):
                    continue
                score = best_score[f"{i} {p}"] + -math.log(float(transition[f"{p} {n}"]))
                if not(f"{i+1} {n}" in best_score) or best_score[f"{i+1} {n}"] < score:
                    best_score[f"{i+1} {n}"] = score
                    best_edge[f"{i+1} {n}"] = f"{i} {p}"
            # print(best_score)
            # print(best_edge)
            tags = []
            next_edge = best_edge[f"{l-1} </s>"]
            while next_edge != "0 <s>":
                # このエッジの品詞を出力に追加
                position, tag = next_edge.split(" ")
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            # print(" ".join(tags))
            f.write(" ".join(tags) + "\n")


if __name__ == "__main__":
    # model = "05-train-output.txt"
    # path = "05-test-input.txt"
    # out_path = "dammy.txt"
    model = "wiki-model.txt"
    path = "wiki-en-test.norm"
    out_path = "my_answer.pos"
    transition, emission = model_read(model)
    with open(path, "r") as f:
        lines = []
        for line in f:
            lines.append(line)
    foward_back_step(transition, emission, lines, out_path)
