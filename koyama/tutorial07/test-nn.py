import numpy as np
import dill

test_path = '../../data/titles-en-test.word'

def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split(' ')
    for word in words:
        if f"UNI:{word}" in ids:
            phi[ids[f"UNI:{word}"]] += 1
    return phi


def forward_nn(net, phi_zero):
    phis = [phi_zero]  # 各層の値
    for i in range(len(net)):
        w, b = net[i]
        # 前の層に基づいて値を計算
        phis.append(np.tanh(np.dot(w, phis[i]) + b))
    return phis  # 各層の結果を返す


if __name__ == '__main__':
    with open('network', 'rb') as f_net, open('ids', 'rb') as f_ids:
        net = dill.load(f_net)
        ids = dill.load(f_ids)
    with open(test_path, 'r') as test_file, open('your_answer', 'w') as answer_file:
        for line in test_file:
            line = line.rstrip()
            phi_zero = create_features(line, ids)
            score = forward_nn(net, phi_zero)[len(net)]
            if score[len(net) - 1] > 0:
                predict = 1
            else:
                predict = -1
            print(predict, file=answer_file)

'''
test.labeled your_answer
Accuracy = 92.419412%
'''
