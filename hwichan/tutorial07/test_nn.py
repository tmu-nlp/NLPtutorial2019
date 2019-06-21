import dill
from collections import Counter
from train_nn import forward_nn


def main():
    with open('net', 'rb') as net_f,\
         open('ids', 'rb') as ids_f:
        net = dill.loads(net_f.read())
        ids = dill.loads(ids_f.read())

    text  = []
    with open('../../data/titles-en-test.word', 'r') as input_file:
         for line in input_file:
            words = line.strip().split(' ')
            test_phi0 = [[0]] * len(ids)
            phi0_dict = Counter(words)
            for word, count in phi0_dict.items():
                if f'UNI:{word}' in ids:
                    test_phi0[ids[f'UNI:{word}']] = [count]
            test_phi = forward_nn(net, test_phi0)
            print(test_phi[len(net)][0][0])
            if test_phi[len(net)][0][0] >= 0:
                text.append('1')
            else:
                text.append('-1')

    with open('test_answer.txt', 'w') as out_file:
        out_file.write('\n'.join(text))


if __name__ == "__main__":
    main()


# Accuracy = 91.817216%
