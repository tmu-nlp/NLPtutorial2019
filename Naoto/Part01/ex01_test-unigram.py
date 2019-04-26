def map_probabilities(s: str) -> {}:
    from collections import defaultdict
    p = defaultdict(lambda: 0)
    for line in s:
        w_and_p = line.split(' ')
        p[w_and_p[0]] = w_and_p[1]
    
    return p

if __name__ == '__main__':
    import sys
    import math
    ipt = open(sys.argv[1], 'r')
    opt = open(sys.argv[2], 'w')
    lam_1 = 0.95
    lam_unk = 1-lam_1
    V = 1000000
    W = 0
    H = 0
    unk = 0
    w_p = map_probabilities(ipt)
    for line in ipt:
        line = line.replace('\n', '').replace(',', '')
        line = line.replace('.', ' /s ')
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


