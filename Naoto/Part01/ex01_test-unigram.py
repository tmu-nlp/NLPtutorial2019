def map_probabilities(s: str) -> {}:
    from collections import defaultdict
    p = defaultdict(lambda: 0)
    for line in s:
        line = line.replace('  ', ' ').replace('\n', '')
        w_and_p = line.split(' ')
        p[w_and_p[0]] = float(w_and_p[1])

    return p

if __name__ == '__main__':
    import sys
    import math
    lam_1 = 0.95
    lam_unk = 1-lam_1
    V = 1000000
    W = 0
    H = 0
    unk = 0
    with open('wiki-en-train_answer.word', 'r') as train, open('wiki-en-test.word', 'r') as test:
    # with open('01-train-answer-mine.txt', 'r') as train, open('01-test-input.txt', 'r') as test:
        w_p = map_probabilities(train)
        for line in test:
            line = line.replace('\n', ' /s').replace(',', '').replace('.', '')
            for i in range(3):
                line = line.replace('  ', ' ')
            words = line.split(" ")
            for word in words:
                W += 1
                p = lam_unk / V
                if w_p[word] != 0: 
                    p += lam_1 * w_p[word]
                else:
                    unk += 1
                H -= math.log2(p)

    print(f"entropy = {H/W}")
    print(f"coverage = {(W - unk)/W}")


