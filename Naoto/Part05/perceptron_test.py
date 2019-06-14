import subprocess
from collections import defaultdict
from perceptron_train import create_features
from perceptron_train import predict_one


def predict(model, input_, my_ans):
    count = 0
    with open(model) as model, open(input_) as f, open(my_ans, "w") as fw:
        w = {}
        for line in model:
            w_k_v = line.split()
            w[w_k_v[0]] = float(w_k_v[1])
        for line in f:
            words = line.split()
            phi = create_features(words)
            # if count == 0:
            #     print(phi)
            #     count += 1
            y_, score = predict_one(w, phi)
            print(f"{y_}\t{line}", file=fw, end="")


if __name__ == "__main__":
    input_ = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/data/titles-en-test.word"
    model = "./model_wiki.txt"
    output_ = "./my_ans.txt"
    predict(model, input_, output_)
