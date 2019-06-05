# train-perceptron.py
from collections import defaultdict

def predict_all(model_file, input_file, output_file):
    with open(model_file) as f_model,\
    open(input_file) as f_input,\
    open(output_file, mode="w") as f_out:
        w = defaultdict(float)
        for line in f_model:
            name, weight = line.split()
            w[name] = weight
        for line in f_input:
            features = create_features(line)
            y = predict_one(w, features)
            # f_out.write(f"{y}\t{line}\n")
            f_out.write(f"{y}\n")

def predict_one(w, features):
    score = 0
    for name, value in features.items():
        if name in w.keys():
            score += float(value) * float(w[name])
    if score >= 0:
        return 1
    else:
        return -1

def create_features(x):
    features = defaultdict(int)
    for word in x.split():
        features["UNI:" + word] += 1
    return features

def update_weights(w, features, y):
    for name, value in features.items():
        w[name] += value * y
    return w

if __name__ == "__main__":
    w = defaultdict(float)
    # input_path = "../../test/03-train-input.txt"
    input_path = "../../data/titles-en-train.labeled"
    output_path = "tutorial05.model"
    for i in range(10):
        with open(input_path) as f:
            for line in f:
                y, x = line.split("\t")
                y = int(y)
                features = create_features(x)
                y_pred = predict_one(w, features)
                if y_pred != y:
                    w = update_weights(w, features, y)
    with open(output_path, mode="w") as fw:
        for key, value in sorted(w.items()):
            fw.write(f"{key} {value}\n")