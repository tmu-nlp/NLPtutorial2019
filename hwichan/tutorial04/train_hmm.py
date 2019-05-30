import sys


def make_map(filename: str) -> dict:
    emit = {}  # tagとwordのペア
    transition = {}  # 一つ前のtag,現在のtagペア
    context = {}  # tagの出現回数
    with open(filename, 'r') as f:
        for line in f:
            previous = "<s>"
            context[previous] = context.get(previous, 0) + 1
            wordtags = line.strip().split(' ')  # word,tag一組ずつで分ける、スペースで分けられている
            for wordtag in wordtags:
                word, tag = wordtag.split('_')[0], wordtag.split('_')[1]  # word, tagで分割
                # 一つ前のtag,現在のtagペアの出現回数を格納
                transition[previous + ' ' + tag] = transition.get(previous + ' ' + tag, 0) + 1  
                context[tag] = context.get(tag, 0) + 1  # tagの出現回数を格納
                emit[tag + ' ' + word] = emit.get(tag + ' ' + word, 0) + 1  # tagとwordのペアの出現回数を格納、wordのtag(品詞)を予測？
                previous = tag
            transition[previous + ' ' + "</s>"] = transition.get(previous + "</s>", 0) + 1

    return emit, transition, context


def write_file(filename: str, text: str):
    with open(filename, 'w') as f:
        f.write(text)


def main():
    emit, transition, context = make_map(sys.argv[1])

    write_text = []
    # transition/context  
    for key, value in transition.items():
        tag = key.split(' ')[0]
        write_text.append('T\t{0}\t{1}'.format(key, value/context[tag]))  # tagペアの出現回数/tagの出現回数

    # emit/context
    for key, value in emit.items():
        tag = key.split(' ')[0]
        write_text.append('E\t{0}\t{1}'.format(key, value/context[tag]))

    write_file(sys.argv[2], '\n'.join(write_text))


if __name__ == "__main__":
    main()

