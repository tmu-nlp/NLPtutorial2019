import sys
import math
from collections import defaultdict


def read_file(filename: str) -> dict:
    weight_map = defaultdict(int)
    with open(filename, 'r') as f:
        for line in f:
            word, weight = line.strip('\n').split(' ')[0], \
                           line.strip('\n').split(' ')[1]
            weight_map[word] = int(weight)

    return weight_map


def write_file(filename: str, text: str):
    with open(filename, 'w') as f:
        f.write(text)


def weight_update(map: dict) -> dict:
    weight_map = defaultdict(int)
    w = 0
    for key, value in map.items():
        y = int(key)
        words = value.split(' ')
        for word in words:
            weight_map[word] += y

    return weight_map


def main():
    weight_map = read_file(sys.argv[1])
    # test : 03-train-answer.txt
    # wiki-train-answer.txt

    text = ''
    with open(sys.argv[2], 'r') as f:
        # test : ../../test/03-train-inpu.txt
        # ../../data//titles-en-test.word
        for line in f:
            y = 0
            words = line.strip('\n').split(' ')
            for word in words:
                y += weight_map[word]

            if y >= 0:
                text += '{}\n'.format(1)
            else:
                text += '{}\n'.format(-1)

    write_file('wiki-test-answer.txt', text)


if __name__ == '__main__':
    main()


# Accuracy = 80.481757%
