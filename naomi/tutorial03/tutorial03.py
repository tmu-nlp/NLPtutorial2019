from collections import defaultdict
from logging import getLogger, StreamHandler, DEBUG
import math
import re


# https://qiita.com/amedama/items/b856b2f30c2f38665701
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


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


def viterbi(text: str, P: dict) -> list:
    
    # 未知語の確率
    unk = 0.05
    # 未知語を含む語彙数
    N = 1e6

    best_edge = defaultdict(lambda: 0)
    best_score = defaultdict(lambda: 0)

    best_edge[0] = None
    best_score[0] = 0

    # 部分文字列の右端のインデックス
    for word_end in range(1, len(text)+1):

        # best_scoreを大きな値に設定
        best_score[word_end] = float('inf')

        # 部分文字列の左端のインデックス
        for word_begin in range(word_end):

            # 部分文字列の取得
            word = text[word_begin:word_end]

            if (word in P) or (len(word) == 1):

                prob = (1 - unk) * P[word] + unk / N

                # エッジの重み（負の対数確率）を計算
                my_score = best_score[word_begin] - math.log2(prob)

                if my_score < best_score[word_end]:
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
    
    # 後ろ向きステップ
    words = []

    next_edge = best_edge[len(best_edge)-1]

    while next_edge is not None:
        # このエッジの部分文字列を追加
        word = text[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
    
    words.reverse()

    return words


def load(file: str) -> dict:       # a	0.0907179533

    # ngramを格納するディクショナリーの用意
    P = defaultdict(lambda: 0)

    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            a = line.rstrip().split()
            if len(a) > 1:
                [ngram, prob] = a
                P[ngram] = float(prob)
    return P


def main():
    # 1-gramの学習データ
    train = '../../data/wiki-ja-train.word'

    # 1-gramモデル
    model = 'mymodel.txt'

    # 読み込むデータ
    inputf = '../../data/wiki-ja-test.word'

    # 分割されたデータ（アウトプット）
    mywakachi = 'myanswer.txt'

    # 1-gramの学習
    trainunigram(train, model)

    # unigram probabilityの読み込み
    P = load(model)

    with open(inputf, 'r', encoding='utf-8') as fin:
        with open(mywakachi, 'w+', encoding='utf-8') as out:
            for line in fin:
                words = viterbi(line.rstrip(), P)
                text = ' '.join(words)
                print(text, file=out)



if __name__ == "__main__":
    main()
