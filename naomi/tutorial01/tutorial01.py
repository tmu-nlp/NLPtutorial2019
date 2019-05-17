import os
from collections import defaultdict
import math


def trainunigram(infile: str, outfile: str):
    with open(infile, encoding='utf-8') as fin:
        with open(outfile, 'w+', encoding='utf-8') as fout:

            lines = fin.readlines()
            # カウントのマップを作る
            wcounts = defaultdict(lambda: 0)

            for line in lines:
                # １行を単語列にしつつ最後に"</s>"を追加
                words = line.replace("\n", " </s>").split(" ")
                
                for word in words:
                    # 辞書に単語と頻度を追加
                    wcounts[word] += 1

            for token, count in sorted(wcounts.items()):

                unigram = count/sum(wcounts.values())

                fout.write(token+'\t{0:.6f}\n'.format(unigram))

    return


def testunigram(modelfile: str, infile: str, outfile: str):

    lambda1 = 0.95
    lambdaunk = 1 - lambda1
    V = 1000000
    unk = 0
    W = 0
    H = 0
    map_prob = defaultdict(lambda: 0)

    with open(modelfile, encoding='utf-8') as fmodel:
        mlines = fmodel.readlines()

        for line in mlines:
            map_prob[line.split()[0]] = line.split()[1]

    with open(infile, encoding='utf-8') as fin:
        with open(outfile, 'w+', encoding='utf-8') as fout:

            ilines = fin.readlines()

            for line in ilines:

                words = line.replace("\n", " </s>").split(" ")

                for word in words:

                    W += 1
                    P = lambdaunk/V

                    print('word='+word)
                    if word in map_prob.keys():
                        P += (lambda1 * float(map_prob[word]))
                    else:
                        unk += 1
                    print('probability={0}'.format(P))
                    H += (-math.log2(P))
                    print('H={0}'.format(H))

            print('entropy='+str(H/W))
            print('coverage='+str((W-unk)/W))
            fout.write('entropy = {0:6f}\n'.format(H/W))
            fout.write('coverage = {0:6f}\n'.format((W-unk)/W))
    return


def main():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    path1 = 'data/wiki-en-train.word'
    path2 = 'data/wiki-en-test.word'
    traindata = os.path.join(THIS_DIR, os.pardir, os.pardir, path1)
    testdata = os.path.join(THIS_DIR, os.pardir, os.pardir, path2)
    modelfile = 'model.txt'
    result = 'entr_coverage.txt'

    trainunigram(traindata, modelfile)
    testunigram(modelfile, testdata, result)


if __name__ == "__main__":
    main()
