import math
lamda1 = 0.95
lamda_unk = 0.05
V = 1000000
W = 0
H = 0
unk_num = 0
prob_dict = {}
model_file = open('./model_file.txt').readlines()
for line in model_file:
    prob_list = line.strip("\n").split(" ")
    prob_dict[prob_list[0]] = prob_list[1]
test_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/01-test-input.txt").readlines()
for line in test_file:
    word_list = line.strip("\n").split(" ")
    word_list.append("</s>")
    for word in word_list:
        W += 1
        P = lamda_unk / V
        if word in prob_dict.keys():
            P += lamda1 * float(prob_dict[word])
        else:
            unk_num += 1
        H += - math.log(P,2) 
print("entropy=",H / W)
print("coverage=",(W-unk_num)/W)


