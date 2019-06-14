from collections import defaultdict

def create_features(sentence):
    phi = defaultdict(lambda:0)
    words = sentence.split(' ')
    for word in words:
        phi['UNI:' + word] += 1
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

def update_weights(w, phi, label):
    for name, value in phi.items():
        w[name] += value * label

input_file = 'test/03-train-input.txt'
input_content = [] 
with open(input_file, 'r') as input_file:
    for line in input_file:
        columm = line.strip().split('\t')
        input_content.append((int(columm[0]), columm[1]))


w = defaultdict(lambda:0)
for i in range(250):
    for label, sentence in input_content:
        phi = create_features(sentence)
        predicted_label = predict_one(w, phi)
        if predicted_label != label:
            update_weights(w, phi, label)

with open('model','w') as model:
    for name, weight in sorted(w.items()):
        model.write(f'{name}\t{weight}\n')