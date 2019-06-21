from discriminative_train import create_features


def predict_one(w, phi: {}):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1


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
            y_ = predict_one(w, phi)
            print(f"{y_}\t{line}", file=fw, end="")


if __name__ == "__main__":
    input_ = "/Users/naoto_nakazawa/komachi_lab/nlptutorial/data/titles-en-test.word"
    model = "./model"
    output_ = "./my_ans"
    c = 10000
    for i in range(10):
        if i % 2 == 0:
            predict(model + str(1/c) + ".txt", input_, output_ + str(1/c) + ".txt")
        else:
            predict(model + str(3/c) + ".txt", input_, output_ + str(3/c) + ".txt")
            c /= 10


'''
c, Accuracy
0.0001, 89.231314%
0.0003, 83.528162%
0.001,  84.307474%
0.003,  79.737868%
0.01,   52.355650%
0.03,   51.895147%
0.1,    51.753454%
0.3,    51.540914%
1.0,    51.540914%
3.0,    51.540914%
'''
