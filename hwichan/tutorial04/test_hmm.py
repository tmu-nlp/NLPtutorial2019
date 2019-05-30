import sys
import math


def make_map(filename: str) -> dict:
    transition = {}
    emit = {}
    possible_tags = {}  # 可能なtag
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            tag_type = line[0]  # tagのペア(T)か、tag,wordのペア(E)か
            context_word = line[1]  # tag,tagまたはtag,wordのペアの種類
            prob = float(line[2])  # 品詞の遷移確率 , 単語の生成確率
            tag = context_word.split(' ')[0]
            possible_tags[tag] = 1  # 学習データに出てくるtag

            if tag_type == "T":
                transition[context_word] = prob
            else:
                emit[context_word] = prob

    return transition, emit, possible_tags


def write_file(filename: str, text: str):
    with open(filename, 'w') as f:
        f.write(text)


def forward_step(words: list, possible_tags: dict, transition: dict, emit: dict) -> dict:
    best_score = {}
    best_edge = {}

    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = None
    v = 100000
    x = 0.95
    for i in range(len(words)):  # i : 何番目の単語か
        for prev in possible_tags:
            for next in possible_tags:
                if ('{} {}'.format(i, prev) in best_score) and ('{} {}'.format(prev, next) in transition):
                    pt = transition['{} {}'.format(prev, next)]  # tag1_teg2の確率
                    pe = x * emit.get('{} {}'.format(next, words[i]), 0) + (1 - x) / v  # tag_wordの確率
                    score = best_score['{} {}'.format(i, prev)] - math.log(pt, 2) - math.log(pe, 2)

                    # ノードのスコアが設定されていなかったら、またはベストスコアより低かったら、更新
                    if ('{} {}'.format(i + 1, next) not in best_score) or \
                            best_score['{} {}'.format(i + 1, next)] > score:  # テキストでは<

                        best_score['{} {}'.format(i + 1, next)] = score
                        best_edge['{} {}'.format(i + 1, next)] = '{} {}'.format(i, prev)

        for tag in possible_tags:
            if ('{} {}'.format(len(words), tag) in best_score) and ('{} {}'.format(tag, '</s>') in transition):
                pt = transition['{} {}'.format(tag, '</s>')]
                score = best_score['{} {}'.format(len(words), tag)] - math.log(pt, 2)

                if ('{} {}'.format(len(words) + 1, '</s>') not in best_score) or \
                        best_score['{} {}'.format(len(words) + 1, '</s>')] > score:
                     
                    best_score['{} {}'.format(len(words) + 1, '</s>')] = score
                    best_edge['{} {}'.format(len(words) + 1, '</s>')] = '{} {}'.format(len(words), tag)

    return best_edge, best_score


def back_step(words: list, best_edge: list) -> list:
    tags = []
    next_edge = best_edge['{} {}'.format(len(words) + 1, '</s>')]

    while True:
        tag = next_edge.split(' ')[1]
        tags.append(tag)
        next_edge = best_edge[next_edge]
        if next_edge == '0 <s>':
            break

    tags.reverse()

    return tags


def main():
    transition, emit, possible_tags = make_map(sys.argv[1])

    tag_list = []
    with open(sys.argv[2], 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            best_edge, best_score = forward_step(words, possible_tags, transition, emit)
            tags = back_step(words, best_edge)
            tag_list.append('\t'.join(tags))

    text = '\n'.join(tag_list)
    print(text)
    # write_file('wiki-test.txt', text)


if __name__ == "__main__":
    main()

