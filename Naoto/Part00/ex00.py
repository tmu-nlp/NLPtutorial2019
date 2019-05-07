def count_each_word(document: str, dic: {}):
    for line in document:
        line = line.replace('\n', '').replace(',', '').replace('.', '')
        words = line.split(" ")
        for word in words:
            if dic[word] != 0:
                dic[word] += 1
            else:
                dic[word] = 1

if __name__ == '__main__':
    from collections import defaultdict
    import sys

    ipt = open(sys.argv[1], "r")
    opt = open(sys.argv[2], "w")
    dic = defaultdict(lambda: 0)
    count_each_word(ipt, dic)
    # sorted(dic.items())

    # output.write('\n'.join(dic))
    for line in sorted(dic):
        opt.write(line + " " +  str(dic[line]) + "\n")

    ipt.close()
    opt.close()
