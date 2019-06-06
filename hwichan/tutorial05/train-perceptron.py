import sys
import math
from collections import defaultdict


def read_file(filename: str) -> dict:
    map = defaultdict()
    with open(filename, 'r') as f:
        for line in f:
            map[line.strip('\n').split('\t')[1]] = line.strip('\n').split('\t')[0]  # テキストのy(予測値), x(文)

    return map


def write_file(filename: str, text: str):
    with open(filename, 'w') as f:
        f.write(text)


def weight_update(map: dict) -> dict:
    weight_map = defaultdict(int)
    for key, value in map.items():
        y = int(value)
        words = key.split(' ')
        for word in words:
            weight_map[word] += y

    return weight_map


def main():
    map = read_file(sys.argv[1])
    # test : ../../test/03-train-input.txt
    # ../../data/titles-en-train.labeled
    weight_map = weight_update(map)

    text = ''
    for key, value in weight_map.items():
        text += '{} {}\n'.format(key, value)

    write_file(sys.argv[2], text)
    # test : 03-train-answer.txt
    # wiki-train-answer.txt


if __name__ == '__main__':
    main()
