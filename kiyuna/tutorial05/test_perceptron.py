'''
重みを読み込み、予測を１行ずつ出力
'''
import os
import sys
import subprocess
from collections import defaultdict
from train_perceptron import create_features, predict_one

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # cd .


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stderr.write("\33[92m" + text + "\33[0m")


def load_model(model_file):
    w = defaultdict(float)
    with open(model_file) as f:
        for line in f:
            key, value = line.split('\t')
            w[key] = float(value)
    return w


def test_perceptron(w, input_file, output_file):
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            sentence = line.rstrip()
            phi = create_features(sentence)
            prediction = predict_one(w, phi)
            print(f'{prediction}\t{sentence}', file=f_out)


if __name__ == '__main__':
    is_test = sys.argv[1:] == ["test"]

    if is_test:
        message("[*] test")
        model_file = './model_test.txt'
        input_file = '../../test/03-train-input.txt'
        output_file = './result_test.labeled'
    else:
        message("[*] wiki")
        model_file = './model_wiki.txt'
        input_file = '../../data/titles-en-test.word'
        output_file = './result_wiki.labeled'

    model = load_model(model_file)
    test_perceptron(model, input_file, output_file)

    if not is_test:
        subprocess.run(
            f'python2 ../../script/grade-prediction.py'\
            f' ../../data/titles-en-test.labeled {output_file}'.split()
        )

    message("[+] Done!")
