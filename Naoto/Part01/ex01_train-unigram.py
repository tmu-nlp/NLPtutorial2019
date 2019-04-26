def count_each_word(document: str, dic: {}) -> int:
    total_count = 0
    for line in document:
        line = line.replace('\n', '').replace(',', '')
        line = line.replace('.', ' /s ')
        for i in range(3):
            line = line.replace('  ', ' ')
        words = line.split(" ")
        for word in words:
            if dic[word] != 0:
                dic[word] += 1
                total_count += 1
            else:
                dic[word] = 1
                total_count += 1
            
    # dic.sort()
    return total_count

if __name__ == '__main__':
    import sys
    from collections import defaultdict
    ipt = open(sys.argv[1], "r")
    opt = open(sys.argv[2], "w")
    words = ""
    dic = defaultdict(lambda: 0)
    total_count = count_each_word(ipt, dic)
    for word in sorted(dic):
        p = dic[word] / total_count
        opt.write(word + "  " + str(p) + "\n")
    
ipt.close()
opt.close()