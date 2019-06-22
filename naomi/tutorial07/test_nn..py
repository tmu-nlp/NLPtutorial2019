from collections import defaultdict


def test_nn(w: dict, lines: list):
    with open('result.txt', 'w+', encoding='utf-8') as f:
        for line in lines:
            line = line.rstrip()
            words = line.split()
            score = 0
            for word in words:
                if word in w:
                    score += w[word]
            if score >= 0:
                result = 1
            else:
                result = -1
            print(f'{result}\t{line}', file=f)


def importmodel(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        w = defaultdict(float)
        for line in f:
            word, weight = line[4:].rstrip().split('\t')
            w[word] = float(weight)
        return w
