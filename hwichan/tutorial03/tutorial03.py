import sys
import math


def create_map(filename: str) -> dict:
    # filenameにはunigram言語モデルを指定
    map = {}
    with open(filename, "r") as f:
        for line in f:
            p = line.strip().split()
            map[p[0]] = float(p[1])

    return map


def write_file(filename: str, text: str):
    with open(filename, "w") as f:
        f.write(text)


def forward_step(line: str, probabilities: dict) -> list:
    x_1 = 0.95
    x_unk = 1 - x_1
    V = 1000000
    best_edge = [None]
    best_score = [0]

    for node in range(1, len(line) + 1):
        best_score.append(10000000000)
        best_edge.append(None)

        for edge in range(node):  # nodeまでのすべてのエッジを探索
            word = line[edge:node]  # edge:node -> path

            if (word in probabilities) or (len(word) == 1):
                prob = x_1 * probabilities.get(word, 0) + x_unk / V
                my_score = best_score[edge] - math.log(prob, 2)

                if my_score < best_score[node]:
                    best_score[node] = my_score
                    best_edge[node] = ((edge, node))

    return best_edge, best_score


def back_step(line: str, best_edge: list) -> list:
    words = []
    next_edge = best_edge[len(best_edge) - 1]

    while True:
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
        if next_edge == None:
            break

    words.reverse()

    return words


def main():
    probabilities = create_map(sys.argv[1])
    words_list = []
    with open(sys.argv[2], 'r') as f:
        for line in f:
            line = line.strip()
            best_edge, best_score = forward_step(line, probabilities)
            words = back_step(line, best_edge)
            words_list.append('\t'.join(words))

    write_file(sys.argv[3], '\n'.join(words_list))


if __name__ == '__main__':
    main()
