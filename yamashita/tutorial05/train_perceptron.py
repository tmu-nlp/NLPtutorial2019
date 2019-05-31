from collections import defaultdict
import argparse
from tqdm import tqdm


def arguments_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str)
    parser.add_argument('epoch_num', type=int)
    return parser.parse_args()


def create_features(sentence):
    phi = defaultdict(int)
    words = sentence.split()
    for word in words:
        phi[f'UNI:{word}'] += 1
    return phi


def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


def train_perceptron(epoch, train_file):
    weight = defaultdict(float)
    for _ in tqdm(range(epoch)):
        with open(train_file, 'r', encoding='utf-8') as i_file:
            for line in i_file:
                label, sentence = line.rstrip().split('\t')
                label = int(label)
                phi = create_features(sentence)
                y_pre = predict_one(weight, phi)
                if y_pre == label:
                    continue
                update_weights(weight, phi, label)

    for key, value in sorted(weight.items()):
        print(f'{key}\t{value:.6f}')


if __name__ == '__main__':
    args = arguments_parse()
    t_file = args.train if args.train else r'../../data/titles-en-train.labeled'
    epoch = args.epoch_num if args.epoch_num else 28

    train_perceptron(epoch, t_file)
