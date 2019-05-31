from collections import defaultdict


def trainu_hiddenmarkov(lines: list):
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
            print(tag)
            # 遷移（前の品詞から今の品詞）の数え上げ
            transition[previous + ' ' + tag] += 1

            # 文脈（品詞の数）の数え上げ
            context[tag] += 1

            # 生成（ある品詞からある単語）の数え上げ
            emit[tag + ' ' + word] += 1

            # 次のステップにおける前の状態（＝今の品詞）をセット
            previous = tag
        # 文末記号を追加
        transition[previous + ' </s>'] += 1

    # 遷移（前の品詞から今の品詞）確率の計算
    for (key, value) in transition.items():
        [previous, word] = key.split()
        print('T {0} {1:.5f}'.format(key, value/context[previous]))
    # 生成（その品詞からその単語）確率の計算
    for (key, value) in emit.items():
        [tag, word] = key.split()
        print('E {0} {1:.5f}'.format(key, value/context[tag]))

    # T <s> X 1.000000
    # T X X 0.333333
    # T X Y 0.666667
    # T Y </s> 0.500000
    # T Y Z 0.500000
    # T Z </s> 1.000000
    # E X a 0.666667
    # E X c 0.333333
    # E Y b 1.000000
    # E Z a 1.000000


def main():

    # 学習データ
    # a_X b_Y a_Z 
    # a_X c_X b_Y

    # (単語_品詞　単語_品詞　…）
    ftrain = '../../test/05-train-input.txt'

    # 学習データのリストを読み込む
    with open(ftrain, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1-gramの学習
    trainu_hiddenmarkov(lines)


if __name__ == "__main__":
    main()
