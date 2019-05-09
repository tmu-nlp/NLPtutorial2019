import sys
import math


def create_map(filename: str) -> dict:
    probabilities = {}
    with open(filename, "r") as f:
        for line in f:
            p = line.strip().split()
            probabilities[p[0]] = float(p[1])

    return probabilities

def result(filename: str, probabilities: dict):
    x_1 = 0.95
    x_unk = 1 - x_1
    V = 1000000
    W = 0
    H = 0
    unk = 0
    with open(filename, "r") as f
        for line in f:
            words = line.strip().split()
            words.append("</s>")
            for w in words: # P : 単語ごとの確率
                P = x_unk / V
                W += 1
                if w in probabilities:
                    P += x_1 * probabilities[w]
                else:
                    unk += 1
                H -= math.log(P, 2)  # H : 底2の負の対数尤度  logであるから、文ごとでも、単語ごとでも同じ
    return  H/W, (W-unk)/W


def main():
    probabilities = create_map(sys.argv[1])
    entropy, coverge = result(sys.argv[2], probabilities)
    print('entropy = ' + str(entropy))
    print('coverge = ' + str(coverge))

if __name__ == '__main__':
    main()
