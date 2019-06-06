if __name__ == "__main__":
    import math
    from collections import defaultdict
    lambda1 = 0.95
    lambda2 = 0.80
    V = 1000000
    W = 0
    H = 0
    probs = defaultdict(lambda: 0)
    model_ = "model.txt"
    test_data_ = "wiki-en-test.word"
    with open(model_, "r") as model:
        for line in model:
            ngram_prob = line.rstrip().split()
            probs[" ".join(ngram_prob[0:-1])] = float(ngram_prob[-1])
    with open(test_data_, "r") as test_data:
        for line in test_data:
            words = ["<s>"] + line.rstrip().split() + ["</s>"]
            W += len(words)
            for i in range(1, len(words)):
                # print(f"lambda1 = {type(lambda1)}")
                # print(f"probs[words[i]] = {type(probs[words[i]])}")
                # print(f"V = {type(V)}")
                # print(f"lambda2 = {type(lambda2)}")
                P1 = lambda1 * probs[words[i]] + (1 - lambda1) / V
                P2 = lambda2 * probs[words[i-1] + " " + words[i]]\
                    + (1 - lambda2) * P1
                H -= math.log2(P2)
    print(f"entropy = {H/W}")
