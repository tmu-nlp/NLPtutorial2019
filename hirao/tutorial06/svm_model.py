# svm_model.py
from collections import defaultdict

def predict_all(model_file, input_file, output_file):
    with open(model_file) as f_model,\
    open(input_file) as f_input,\
    open(output_file, mode="w") as f_out:
        w = defaultdict(lambda: defaultdict(int))
        for line in f_model:
            name, weight = line.split()
            weight = float(weight)
            w["w"][name] = weight
        for index, line in enumerate(f_input):
            features = create_features(line)
            y = predict_one(w, features, index)
            # f_out.write(f"{y}\t{line}\n")
            f_out.write(f"{y}\n")

def predict_one(w, features, iter_):
    score = 0
    for name, value in features.items():
        if name in w["w"].keys():
            score += float(value) * float(getw(w, name, iter_))
    if score >= 0:
        return 1
    else:
        return -1

def create_features(x):
    features = defaultdict(int)
    for word in x.split():
        features["UNI:" + word] += 1
    return features

def sign(v):
    if v > 0:
        return 1
    elif v < 0:
        return -1
    else:
        return 0

def getw(w, name, iter_, c=0.0001):
    if iter_ != w["last"][name]:
        c_size = c * (iter_ - w["last"][name])
        if abs(w["w"][name]) <= c_size:
            w["w"][name] = 0
        else:
            w["w"][name] -= sign(w["w"][name]) * c_size
        w["last"][name] = iter_
    return w["w"][name]

def update_weights(w, features, y, c=0.0001):
    for name, value in features.items():
        w["w"][name] += value * y
    return w

if __name__ == "__main__":
    w = defaultdict(lambda: defaultdict(int))
    # input_path = "../../test/03-train-input.txt"
    input_path = "../../data/titles-en-train.labeled"
    output_path = "tutorial06.model"
    # 学習回数
    for i in range(10):
        with open(input_path) as f:
            for index,line in enumerate(f):
                y, x = line.split("\t")
                y = int(y)
                features = create_features(x)
                y_pred = predict_one(w, features, index)
                if y_pred != y:
                    w = update_weights(w, features, y)
    with open(output_path, mode="w") as fw:
        for key, value in sorted(w["w"].items()):
            fw.write(f"{key} {value}\n")
    model_path = "tutorial06.model"
    test_path = "../../data/titles-en-test.word"
    output_path = "tutorial06.result"
    predict_all(model_path, test_path, output_path)
    # Accuracy = 93.057032%