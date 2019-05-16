training_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/01-train-input.txt").readlines()
print(training_file)
counts = {}
total_count = 0
for line in training_file:
    word_list = line.strip("\n").split(" ")
    word_list.append("</s>")
    for word in word_list:
        counts.setdefault(word,0)
        counts[word] += 1
        total_count += 1
model_file = open('./model_file.txt', 'w')
for word,count in counts.items():
    prob = float(counts[word] / total_count)
    model_file.write("{0} {1}\n".format(word,prob))
model_file.close()
