from collections import defaultdict
from tqdm import tqdm

def create_feats(words: list, tags: list) -> defaultdict:
    phi = defaultdict(int)
    for i in range(len(tags) + 1):
        first = tags[i - 1] if i == 0 else "<s>"
        next_ = tags[i] if i != len(tags) else "</s>"

        phi[f"T {first} {next_}"] += 1
        if i == len(tags):
            break
        phi[f"E {tags[i]} {words[i]}"] += 1

    return phi

def viterbi(words, possible_tags, weights, trans):
    l = len(words)
    best_score, best_edge = {"0 <s>": 0}, {"0 <s>": None}

    for i in range(l):
        for prev in possible_tags:
            key = f"{i} {prev}"
            for nxt in possible_tags:
                t_key, e_key = f"{prev} {nxt}", f"E {nxt} {words[i]}"
                if key not in best_score or t_key not in trans:
                    continue

                score = best_score[key] + weights["T " + t_key] + weights[e_key]
                n_key = f"{i+1} {nxt}"
                if n_key not in best_score or best_score[n_key] < score:
                    best_score[n_key], best_edge[n_key] = score, key
    # 文末
    for tag in possible_tags:
        if not trans[f"{tag} </s>"]:
            continue

        key, t_key = f"{l} {tag}", f"{tag} </s>"
        score = best_score[key] + weights["T " + t_key]
        n_key = f"{l+1} </s>"
        if n_key not in best_score or best_score[n_key] < score:
            best_score[n_key], best_edge[n_key] = score, key
    
    tags, next_edge = [], best_edge[f"{l+1} </s>"]
    while next_edge != "0 <s>":
        _, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags = tags[::-1]

    return tags

def update_weights(w, phi_prime, phi_hat):
    for k, v in phi_prime.items():
        w[k] += v
    for k, v in phi_hat.items():
        w[k] -= v

def load_trainfile(epoch=1):
    # trainfile = "../files/test/05-train-input.txt"
    trainfile = "../files/data/wiki-en-train.norm_pos"
    for _ in tqdm(range(epoch), desc="epochs"):
        for line in tqdm(open(trainfile, encoding="utf-8")):
            words_tags = line.strip().split()
            words, tags = [], []

            for word_tag in words_tags:
                word, tag = word_tag.split("_")
                words.append(word)
                tags.append(tag)
            yield words, tags

def preprocesser():
    trans = defaultdict(int)
    possible_tags = {"<s>", "</s>"}

    for _, tags in load_trainfile():
        prevs, nexts = ["<s>"] + tags, tags + ["</s>"]
        for prv, nxt in zip(prevs, nexts):
            trans[f"{prv} {nxt}"] += 1
        possible_tags.update(tags)

    return trans, possible_tags

def train():
    transition, possible_tags = preprocesser()
    weights = defaultdict(int)

    for words, tags_prime in load_trainfile(epoch=2):
        tags_hat = viterbi(words, possible_tags, weights, transition)
        phi_prime = create_feats(words, tags_prime)
        phi_hat = create_feats(words, tags_hat)

        update_weights(weights, phi_prime, phi_hat)

    return transition, possible_tags, weights

def load_testfile():
    # testfile = "../files/test/05-test-input.txt"
    testfile = "../files/data/wiki-en-test.norm"
    for line in open(testfile, "r", encoding="utf-8"):
        words = line.strip().split()
        yield words

def test(trainsition, possible_tags, weights):
    with open("ans2", "w", encoding="utf-8") as f:
        for words in load_testfile():
            y_hat = viterbi(words, possible_tags, weights, trainsition)
            print(" ".join(y_hat), file=f)


if __name__ == "__main__":
    tr, ps, w = train()
    test(tr, ps, w)