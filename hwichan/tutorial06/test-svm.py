import sys
from collections import defaultdict


def create_map(filename:str):
    weight_map = defaultdict(lambda:0)
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            word = line[0]
            weight = float(line[1])
            weight_map[word] = weight

    return weight_map


def main():
    weight_map = create_map('train_answer_margin0.txt')

    text = ''
    with open('../../data/titles-en-test.word', 'r') as f:
        for line in f:
            words = line.strip().split(' ')

            prob = 0
            for word in words:
                prob += weight_map[word]

            if prob >= 0:
                y = 1
            else:
                y = -1

            text += f'{y}\n'

    with open('test_answer_margin0.txt', 'w') as f:
        f.write(text)


if __name__ == "__main__":
    main()


# margin = 20, c = 0.001 -> Accuracy = 91.640099%
# margin = 0, c = 0.001 -> Accuracy = 91.250443%
