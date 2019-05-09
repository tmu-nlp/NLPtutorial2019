import sys

def read_file(filename: str) -> list:
    words_list = []
    with open(filename, "r") as file:
        for line in file:
            words_list +=  line.strip().split() # 一列ずつ読み込み単語で分割し格納
            words_list.append("</s>") # 一列（一文）読み込んだら文末記号をappend

    return words_list

def write_file(filename: str, text: str):
    with open(filename, "w") as file:
        file.write(text)


def main():
    word_count = {}  # key : word, value : count
    all_count = 0  # 単語の総数
    words_list = read_file(sys.argv[1])  # 単語のlist
    for word in words_list:
        word_count[word] = word_count.get(word, 0) + 1  #
        all_count += 1

    text = ''
    for word, count in word_count.items():
        text += '{0}\t{1}\n'.format(word, count/all_count)  # count/all_count

    write_file(sys.argv[2], text)

if __name__ == '__main__':
    main()
