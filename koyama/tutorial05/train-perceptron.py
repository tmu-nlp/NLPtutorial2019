from collections import defaultdict

def create_features(x):
    phi = defaultdict(lambda: 0)
    words = x.split(' ')
    for word in words:
        phi["UNI:" + word] += 1
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

w = defaultdict(lambda: 0)
with open('../../data//titles-en-train.labeled', 'r') as input_file, open('model-file.txt', 'w') as model_file:
    for line in input_file:
        line = line.strip().split('\t')
        x = line[1]
        y = int(line[0])
        phi = create_features(x)
        y_prime = predict_one(w, phi)
        if y_prime != y:
            update_weights(w, phi, y)
    for name, value in sorted(w.items()):
        print(f'{name} {value}', file=model_file)
