from collections import defaultdict
from tqdm import tqdm

epoch = 70
margin = 20
c = 0.0001


def create_features(sentence):
    phi = defaultdict(lambda: 0)
    words = sentence.split(' ')
    for word in words:
        phi['UNI:' + word] += 1
    for i in range(len(words) - 1):
        phi['BI:' + words[i] + ' ' + words[i + 1]] += 1
    return phi


def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score > 0:
        return 1
    else:
        return -1


def get_val(w, phi, label, iter_, last):
    val = 0
    for name, value in phi.items():
        if name in w:
            val += value * getw(w, name, c, iter_, last)
    return val * label


def update_weights(w, phi, label):
    for name, value in phi.items():
        w[name] += value * label


def getw(w, name, c, iter_, last):
    if iter_ != last[name]:
        c_size = c * (iter_ - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            if w[name] >= 0:
                w[name] -= c_size
            else:
                w[name] += c_size
        last[name] = iter_
    return w[name]

if __name__ == '__main__':
    input_path = '../../data/titles-en-train.labeled'
    input_ = []
    with open(input_path, 'r') as input_file:
        for line in input_file:
            columm = line.strip().split('\t')
            input_.append((int(columm[0]), columm[1]))
    w = defaultdict(lambda: 0)
    last = defaultdict(lambda: 0) 
    for _ in tqdm(range(epoch)):
        for i, (label, sentence) in enumerate(input_):
            phi = create_features(sentence)
            val = get_val(w, phi, label, i, last)
            if val <= margin:
                update_weights(w, phi, label)
    with open('model', 'w') as model:
        for name, weight in sorted(w.items()):
            model.write(f'{name}\t{weight}\n')
    w = {}
    with open('model', 'r') as model:
        for line in model:
            columm = line.strip().split('\t')
            w[columm[0]] = int(float(columm[1]))

    test_path = '../../data/titles-en-test.word' 
    with open(test_path, 'r') as test_file:
        for line in test_file:
            line = line.rstrip()
            phi = create_features(str(line))
            predicted_label = predict_one(w, phi)
    print(predicted_label)