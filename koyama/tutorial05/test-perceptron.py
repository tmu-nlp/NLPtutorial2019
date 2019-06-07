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

def predict_all(model_file, input_file):
    w = {}
    with open(model_file, 'r') as m_file:
        for line in m_file:
            line = line.strip().split(' ')
            name = line[0]
            value = int(line[1])
            w[name] = value
    with open(input_file, 'r') as i_file, open('your_answer.txt', 'w') as answer_file:
        for line in i_file:
            phi = create_features(line)
            y_prime = predict_one(w, phi)
            print(y_prime, file=answer_file)

predict_all('model-file.txt', '../../data/titles-en-test.word')

#精度
#t.labeled your_answer.txt
#Accuracy = 90.860786%
