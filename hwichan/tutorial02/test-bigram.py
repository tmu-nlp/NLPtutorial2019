import sys
import math


def create_map(filename: str) -> dict:
    map = {}
    with open(filename, "r") as f:
        for line in f:
            p = line.strip().split('\t')
            map[p[0]] = float(p[1])

    return map

def result(filename: str, probabilities: dict, witten_bell: dict):
    # x_1 = 0.95
    # x_unk = 1 - x_1
    V = 1000000
    W = 0
    H = 0
    unk = 0
    with open(filename, "r") as f:
        for line in f:
            words = line.strip().split()
            words.insert(0, "<s>")
            words.append("</s>")
            W += 1  # <s>を足したので単語の総数 W に加算
            for i in range(len(words)-1):
                w = words[i] + ' '+words[i + 1]  # bigramの生成
                p1 = 0.95 * probabilities.get(words[i+1], 0) + (1 - 0.95)/V
                lam_2 = witten_bell.get(words[i], 0)
                p2 = (1 - lam_2) * p1
                W += 1
                if w in probabilities:
                    p2 += lam_2 * probabilities[w]
                else:
                    unk += 1

                if p2 == 0:
                    print(words[i] + str(x_2))
                    break
                H -= math.log(p2, 2)  # H : 底2の負の対数尤度  対数であるから、文ごとでも、単語ごとでも同じ
    return  H/W, (W-unk)/W


def main():
    probabilities = create_map('train-answer.txt')
    witten_bell = create_map('witten_bell.txt')

    # i = -1
    # for w in probabilities:
    #     if 'a' in w:
    #         i += 1
    # print(i)

    entropy, coverge = result(sys.argv[1], probabilities, witten_bell)
    print('entropy = ' + str(entropy))
    print('coverge = ' + str(coverge))
    # print(2**entropy)

if __name__ == '__main__':
    main()
