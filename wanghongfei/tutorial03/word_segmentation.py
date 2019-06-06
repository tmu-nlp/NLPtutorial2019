import math
#load unigram
unigram_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/04-model.txt", "r").readlines()
unigram = {}
for line in unigram_file:
    prob_list = line.strip("\n").split("\t")
    unigram[prob_list[0]] = prob_list[1]
print(unigram)
#input sentence
input = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/04-input.txt", "r").readlines()
for line in input:
        line = line.strip("\n")
        best_edge = [None] * (len(line)+1)
        best_score = [0] * (len(line)+1)
        for word_end in range(1,len(line)+1):
                best_score[word_end] = 1e10
                for word_begin in range(word_end):
                        word = line[word_begin:word_end]
                        if word in unigram.keys() or len(word) == 1:
                                prob = float(unigram[word])
                                my_score = float(best_score[word_begin]) - math.log(prob,2)
                                if my_score < best_score[word_end]:
                                        best_score[word_end] = my_score
                                        best_edge[word_end] = (word_begin, word_end)
        print(best_edge)
        print(best_score)
        words = []
        next_edge = best_edge[-1]
        while next_edge != None:
                word = line[next_edge[0]:next_edge[1] ]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(words)
                

