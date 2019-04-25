import os
def word_count(word_file):
    word_list = word_file.replace("\n"," ").split(" ")
    word_count = {}
    for word in word_list:
        word_count.setdefault(word, 0)
        word_count[word] += 1
    print(word_count)

test_input = "this is a pen\nthis pen is my pen"
word_count(test_input)
test_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/data/wiki-en-train.word")
test_content = test_file.read()
word_count(test_content)






