import sys
import math


# words_list -> key:単語（bi-gram, uni-gram） value:単語の出現数
# context_counts -> key:一つ前にある単語 value:その出現数
def create_map(filename: str) -> dict:
    words_list = {}
    context_counts = {}
    with open(filename, "r") as f:
        for line in f:
            words = line.strip().split()
            words.insert(0, "<s>")
            words.append("</s>")
            words_list["<s>"] = words_list.get("<s>", 0) + 1
            for n in range(len(words)-1):
                words_list[words[n]+' '+words[n+1]] = words_list.get(words[n]+' '+words[n+1], 0) + 1  # bi-gramの分子
                context_counts[words[n]] = context_counts.get(words[n], 0) + 1  # bi-gramの分母
                words_list[words[n+1]] = words_list.get(words[n+1], 0) + 1  # uni-gramの分子 二単語目のみカウント
                context_counts[""] = context_counts.get("", 0) + 1  # uni-gramの分子 単語の総数
    return words_list, context_counts

def write_file(filename: str, text: str):
    with open(filename, "w") as f:
        f.write(text)

def main():
    words_list, context_counts = create_map(sys.argv[1])
    print(words_list)
    print(context_counts)
    witten_bell = {}

    P = ''
    for w in words_list:
        w1 = w.strip().split()[:-1]  # bigramの場合一単語目、unigramの場合空文字
        # bigramの場合、bigramの出現回数/一単語目の出現回数
        # unigramの場合、単語の出現回数/単語の総数(空文字は単語の総数)
        P += w+'\t'+str(words_list[w]/context_counts[''.join(w1)])+'\n'
    print(P)

    # witten_bell
    W = ''
    for c, count in context_counts.items():  # c : 出現単語, count : 出現回数
        if c == '':
            continue
        u = 0  # 二単語目の異なり数
        for w in words_list:  # words_listのkeyは同じbigramを繰り返さないから二単語目の異なり数を取ってこれる
            if c == ''.join(w.strip().split()[:-1]):  # bigramの１単語目と合致したら
                u += 1

        W += c + '\t' + str(1 - u / (u + count)) + '\n'  # 1 - 二単語目の異なり数/(二単語目の異なり数 ＋ 一単語目の出現回数)
    print(W)
    write_file(sys.argv[2], P)
    write_file(sys.argv[3], W)

if __name__ == '__main__':
    main()
