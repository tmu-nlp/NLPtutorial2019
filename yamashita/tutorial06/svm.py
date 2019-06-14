from collections import defaultdict
from tqdm import tqdm


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


def predict_one_(w, phi, i, last):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * getw(w, name, i, last)
    return score


def load_weights(weights_file):
    w = defaultdict(float)
    with open(weights_file, 'r', encoding='utf-8') as w_file:
        for line in w_file:
            key, value = line.rstrip().split('\t')
            w[key] = float(value)
    return w


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


def sign(val):
    return 1 if val >= 0 else 0


def getw(w, name, iter, last, c=0.0001):
    if iter != last[name]:
        c_size = c*(iter - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]


def train_svm(epoch, input_path, output_path, margin=20):
    w = defaultdict(float)
    last = defaultdict(int)
    for _ in tqdm(range(epoch)):
        with open(input_path, 'r', encoding='utf-8') as i_file:
            for i, line in enumerate(i_file):
                label, sentence = line.rstrip().split('\t')
                label = int(label)
                phi = create_features(sentence)
                val = label * predict_one_(w, phi, i, last)
                if val > margin:
                    continue
                update_weights(w, phi, label)

    with open(output_path, 'w', encoding='utf-8') as o_file:
        for key, value in sorted(w.items()):
            print(f'{key}\t{value}', file=o_file)


def test_svm(model_path, input_path, output_path):
    w = load_weights(model_path)
    with open(input_path, 'r', encoding='utf-8') as i_file, open(output_path, 'w', encoding='utf-8') as o_file:
        for line in i_file:
            line = line.strip()
            phi = create_features(line)
            pre = predict_one(w, phi)
            print(f'{pre}\t{line}', file=o_file)


def main():
    train_input_path = '../../data/titles-en-train.labeled'
    train_output_path = 'svm_model'
    epoch_num = 10
    train_svm(epoch_num, train_input_path, train_output_path)

    modelpath = 'svm_model'
    test_input_path = '../../data/titles-en-test.word'
    test_output_path = 'result_svm'
    test_svm(modelpath, test_input_path, test_output_path)


if __name__ == '__main__':
    main()
