from train_svm import train_svm
from test_svm import test_svm, importmodel


def main():

    # 学習データ
    ftrain = '../../data/titles-en-train.labeled'

    # 学習モデル
    fmodel = 'mymodel.txt'

    # データ
    fdata = '../../data/titles-en-test.word'

    # 学習データのリストを読み込む
    train_svm(ftrain, fmodel)
    w = importmodel(fmodel)

    with open(fdata, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        test_svm(w, lines)


if __name__ == "__main__":
    main()