from test_perceptron import test_perceptron, importmodel
from train_perceptron import train_perceptron


def main():

    # 学習データ
    ftrain = '../../test/03-train-input.txt'
    ftrain = '../../data/titles-en-train.labeled'

    # 学習モデル
    fmodel = 'mymodel.txt'

    # データ
    fdata = 'data.txt'
    fdata = '../../data/titles-en-test.word'

    # 学習データのリストを読み込む
    train_perceptron(ftrain, fmodel)
    w = importmodel(fmodel)

    with open(fdata, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        test_perceptron(w, lines)


if __name__ == "__main__":
    main()