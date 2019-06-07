import argparse
from collections import defaultdict
from train_perceptron import create_features, predict_one


def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str)
    parser.add_argument('input_', type=str)
    return parser.parse_args()


def load_weights(weights_file):
    w = defaultdict(float)
    with open(weights_file, 'r', encoding='utf-8') as w_file:
        for line in w_file:
            key, value = line.rstrip().split('\t')
            w[key] = float(value)
    return w


def test_perceptron(w, input_file):
    with open(input_file, 'r', encoding='utf-8') as i_file:
        for line in i_file:
            snetence = line.rstrip()
            phi = create_features(snetence)
            prediction = predict_one(w, phi)
            print(f'{prediction}\t{snetence}')


if __name__ == '__main__':
    args = arguments_parse()
    weights_file = args.weights if args.weights else r'uni_weights.txt'
    input_file = args.input_ if args.input_ else r'../../data/titles-en-test.word'

    w = load_weights(weights_file)
    test_perceptron(w, input_file)
