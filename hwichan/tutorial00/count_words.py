import sys
import re

file = open(sys.argv[1], "r")

list = {}
for line in file: #一行ずつはいってくる
    line = line.strip() #まず改行を消す
    line =  re.split('[., ]',line) #.,は単語に入れないことにする
    line = [x for x in line if x is not ""]
    # print(line)

    for i in line:
        if i in list: #すでに単語が格納されていれば１加算
            list[i] += 1
        else:         #単語がなかったら単語をキーとして連想配列に格納
            list[i] = 1

file.close()

for i in list:
    print(i+"\t"+str(list[i]))

print("単語の異なり数 : " + str(len(list)))
