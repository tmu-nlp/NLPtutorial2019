from collections import defaultdict
from typing import Union
import math
import itertools


def train_hmm(lines: list, path: str):

    # 生成を格納する
    emit = defaultdict(lambda: 0)
    # 遷移を格納する
    transition = defaultdict(lambda: 0)
    # 文脈の頻度
    context = defaultdict(lambda: 0)

    # １文ずつ
    for line in lines:
        # 最初の品詞（＝隠れた状態）を文頭記号とおく
        previous = '<s>'

        context[previous] += 1
        # 改行コードを取り除き、文末記号を追加
        line = line.rstrip('\n')

        # 単語_品詞（＝表層の状態＿隠れた状態）を要素とするリストに分割
        wordtags = line.split(' ')

        # 単語＿品詞ずつ
        for wordtag in wordtags:

            # 単語と品詞に分ける
            [word, tag] = wordtag.split('_')

            # 遷移（前の品詞から今の品詞）の数え上げ
            transition[previous + ' ' + tag] += 1

            # 文脈（品詞の数）の数え上げ
            context[tag] += 1

            # 生成（ある品詞からある単語）の数え上げ
            emit[tag + ' ' + word] += 1

            # 次のステップにおける前の状態（＝今の品詞）をセット
            previous = tag

        # 文の終わりに文末記号を追加
        transition[previous + ' </s>'] += 1

    with open(path, 'w+', encoding='utf-8') as f:

        # 遷移（前の品詞から今の品詞）確率の計算
        for (key, value) in transition.items():
            [previous, word] = key.split()
            print('T {0} {1:.5f}'.format(key, value/context[previous]), file=f)
        # 生成（その品詞からその単語）確率の計算
        for (key, value) in emit.items():
            [tag, word] = key.split()
            print('E {0} {1:.5f}'.format(key, value/context[tag]), file=f)


def importmodel(lines: list) -> Union[dict, dict, dict, dict]:
    '''
    lines: モデルの文
    p_emit: 生成確率を格納する
    p_transition: 遷移確率を格納する
    possible_tags: タグを格納する

    '''

    # 生成確率（品詞から単語）を格納する
    p_emit = defaultdict(lambda: 0)
    # 遷移確率（前の品詞から今の品詞）を格納する
    p_transition = defaultdict(lambda: 0)
    # タグ（品詞）を格納する
    possible_tags = defaultdict(lambda: 0)
    # トークンを格納する
    possible_tokens = defaultdict(lambda: 0)


    # モデル（生成確率、遷移確率の）の読み込み
    for line in lines:
        (TE, key1, key2, val) = line.rstrip().split()

        possible_tags[key1] += 1
        possible_tokens[key2] += 1

        if TE == 'T':
            p_transition[key1 + ' ' + key2] = float(val)
        elif TE == 'E':
            p_emit[key1 + ' ' + key2] = float(val)
        
    return p_emit, p_transition, possible_tags, possible_tokens


def test_hmm(lines: list, 
             p_emit: dict, 
             p_transition: dict, 
             possible_tags: dict,
             possible_tokens: dict,
             l1: float) -> list:

    # 最後に返す品詞列を格納するリストを用意
    tags_list = []

    # 未知語を含んだ語彙数
    V = 1e6

    # 未知語の数
    Unk = 0

    # １文ずつ
    for line in lines:

        # 単語に分割
        words = line.split()
        l = len(words)

        # best_score['1 NN']：１つ目の単語がNNとタグ付けされた時のベストスコア
        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)

        # 文頭：0語目の単語のタグは<s>として、ベストスコアの初期値を与える
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None

        # wordごと。i: 0, 1, ... l-1
        for i in range(l):

            # 未知語のカウント
            if possible_tokens[words[i]] == 0:
                Unk += 1

            for prv, nxt in itertools.product(possible_tags.keys(), 
                                              possible_tags.keys()):

                if '{0} {1}'.format(i, prv) not in best_score \
                   or (prv + ' ' + nxt) not in p_transition:
                    continue

                # 遷移確率
                Pt = p_transition[prv + ' ' + nxt]

                # 未知語を含むときの生成確率
                Pe = l1 * p_emit[nxt + ' ' + words[i]] + (1-l1)/V

                # スコアの計算
                score = (best_score['{0} {1}'.format(i, prv)]
                         - math.log2(Pt)
                         - math.log2(Pe))

                # ベストスコアのチェック（小さいほどよい）
                if (str(i+1)+' '+nxt) in best_score \
                   and best_score[str(i+1)+' '+nxt] < score:
                    continue

                # ベストスコアの更新
                best_score[str(i+1)+' '+nxt] = score
                best_edge[str(i+1)+' '+nxt] = '{0} {1}'.format(i, prv)

        # # 文末記号への遷移を考える
        for tag in possible_tags.keys():

            if '{0} {1}'.format(l, tag) not in best_score \
               or tag + ' </s>' not in p_transition:
                continue
            
            # 遷移確率
            Pt = p_transition[tag+ ' </s>']

            # 未知語を含むときの生成確率
            Pe = l1 * p_emit[tag+ ' </s>'] + (1-l1)/V

            # スコアの計算
            score = (best_score['{0} {1}'.format(l, tag)]
                     - math.log2(Pt)
                     - math.log2(Pe))
            
            # ベストスコアのチェック（小さいほどよい）
            if str(l+1)+' </s>' in best_score \
               and best_score[str(l+1)+' </s>'] < score:
                continue

            # ベストスコアの更新
            best_score[str(l+1)+' </s>'] = score
            best_edge[str(l+1)+' </s>'] = '{0} {1}'.format(l, tag)

        tags = []

        next_edge = best_edge[str(l+1)+' </s>']

        while next_edge != '0 <s>':
            # このエッジの品詞を出力に追加
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]

        # 順番を入れ替える
        tags = tags[::-1]
        tags_list.append(' '.join(tags))

    print(f'words in the model: {len(possible_tokens)}')
    print(f'Unkown words: {Unk}')
    return tags_list


def main():

    # 学習データ　(単語_品詞　単語_品詞　…）
    # a_X b_Y a_Z 
    # a_X c_X b_Y

    # 学習データ
    ftrain = '../../test/05-train-input.txt'
    ftrain = '../../data/wiki-en-train.norm_pos'

    # 学習モデル
    fmodel = '../../test/05-train-answer.txt'
    fmodel = 'mymodel.txt'

    # データ
    fdata = '../../data/wiki-en-test.norm'

    # 結果
    fresult = 'result.txt'

    # 学習データのリストを読み込む
    with open(ftrain, 'r', encoding='utf-8') as ft:
        lines = ft.readlines()
        train_hmm(lines, fmodel)

    with open(fmodel, 'r', encoding='utf-8') as fm:
        lines = fm.readlines()
        (p_emit, p_transition, possible_tags, possible_tokens) = importmodel(lines)

    with open(fdata, 'r', encoding='utf-8') as fin, open(fresult, 'w+', encoding='utf-8') as fout:
        lines = fin.readlines()
        results = test_hmm(lines, p_emit, p_transition, possible_tags, possible_tokens, 0.9)
        for result in results:
            print(result, file=fout)


if __name__ == "__main__":
    main()

# perl gradepos.pl ../../data/wiki-en-test.pos result.txt

# V: 1e6, l1: 0.9
# Accuracy: 90.82% (4144/4563)

# Most common mistakes:
# NNS --> NN      45
# NN --> JJ       27
# NNP --> NN      22
# JJ --> DT       22
# JJ --> NN       12
# VBN --> NN      12
# NN --> IN       11
# NN --> DT       10
# NNP --> JJ      8
# VBN --> JJ      7