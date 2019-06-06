'''
1-gram モデルを読み込み、エントロピーとカバレージを計算
'''
import os
import sys
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text):
    print("\33[92m" + text + "\33[0m")


def load_model(path):
    model = {}
    with open(path) as f:
        for line in f:
            word, prob = line.split('\t')
            model[word] = float(prob)
    return model


def test_unigram(p_ml, test):
    # P(w_i) = λ_1 P_ML(w_i) + (1 − λ_1) / V
    λ_1 = 0.95
    λ_unk = 1 - λ_1     # 未知語確率
    V = 1000000         # 未知語を含む語彙数
    W = 0               # 単語数
    H = 0               # 負の底 2 の対数尤度
    unk = 0             # 未知語の数
    with open(test) as f:
        for line in f:
            words = line.rstrip().split() + ["</s>"]     # append EOS
            for wi in words:
                W += 1
                P = λ_unk / V
                if wi in model:
                    P += λ_1 * p_ml[wi]
                else:
                    unk += 1
                H += -math.log2(P)
    entropy = H / W
    coverage = (W - unk) / W
    return entropy, coverage


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        path = './model_test.txt'
        test = '../../test/01-test-input.txt'
    else:
        message("[*] wiki")
        path = './model_wiki.txt'
        test = '../../data/wiki-en-test.word'

    model = load_model(path)
    entropy, coverage = test_unigram(model, test)
    print(f"entropy  = {entropy:f}")
    print(f"coverage = {coverage:f}")

    message("[+] Finished!")
