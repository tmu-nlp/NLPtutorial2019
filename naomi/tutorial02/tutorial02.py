from collections import defaultdict
import math

# train-bigram: 2-gram モデルを学習


def trainbigram(infile: str, outfile: str):

    with open(infile, encoding='utf-8') as fin:

        counts = defaultdict(lambda: 0)
        context_counts = defaultdict(lambda: 0)

        # initialize array of words
        words = []

        for line in fin:
            # 文末に'<s>'を入れつつ単語に分割
            words = line.replace('\n', ' <s>').split(' ')
            # 文頭に'<s>'を挿入
            words.insert(0, '<s>')

            for i in range(1, len(words)):
                # bigramの分子
                counts[words[i-1]+' '+words[i]] += 1
                # bigramの分母
                context_counts[words[i-1]] += 1
                # 1gramの分母
                counts[words[i]] += 1
                # 1gramの分母
                context_counts[''] += 1

        with open(outfile, 'w', encoding='utf-8') as fout:
            for ngram, count in sorted(counts.items()):
                # split ngram into an array of words
                tokens = ngram.split(' ')

                if ' ' not in ngram:
                    context = ''
                else:
                    context = tokens[0]

                probability = count / context_counts[context]
                print('{0}\t{1:.5f}'.format(ngram, probability), file=fout)

    return None

# test-bigram: 2-gram モデルに基づいて評価データのエントロピーを計算


def testbigram(modelf: str, testf: str):

    lambda1 = 0.95
    lambda2 = 0.8

    V = 1000000
    W = 0   # テストしたいファイルに含まれるトークン数の初期化
    H = 0   # エントロピーの初期化

    probs = defaultdict(lambda: 0)

    # モデルファイルの読み込み
    with open(modelf, 'r', encoding='utf-8') as f:
        for line in f:
            [ngram, p] = line.rstrip('\n').split('\t')
            probs[ngram] = float(p)

    w = []

    # テストしたいファイルの読み込み
    with open(testf, 'r', encoding='utf-8') as f:
        for line in f:
            # 文末に'<s>'を入れつつ単語に分割
            words = line.replace('\n', ' <s>').split(' ')
            # 文頭に'<s>'を挿入
            words.insert(0, '<s>')
            w.extend(words)
        print(w)

    for i in range(1, len(w)):
        P1 = lambda1 * probs[w[i]] + (1 - lambda1) / V
        P2 = lambda2 * probs[w[i-1]+' '+w[i]] + (1 - lambda2) * P1

        H += -math.log2(P2)
        W += 1

    print('entropy= {0}'.format(H/W))


def main():

    # 読み込む学習データ
    trainf = '../../data//wiki-en-train.word'

    # 作成するモデル
    modelf = 'wikimodel.txt'

    # テストするデータ
    testf = '../../data/wiki-en-test.word'

    # wikiのデータで学習、model作成
    trainbigram(trainf, modelf)

    # 作成したモデル（ユニグラム、バイグラムの確率）を使ってエントロピーを計算
    testbigram(modelf, testf)


if __name__ == "__main__":
    main()
