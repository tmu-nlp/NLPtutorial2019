training_file = open("/Users/hongfeiwang/desktop/nlptutorial-master/test/02-train-input.txt", "r").readlines()
counts = {}
context_counts = {}
for line in training_file:
        words = line.split()
        words.insert(0, "<s>")
        words.append("</s>")
        print(words)
        for i in range(1,len(words)):
            counts.setdefault(words[i-1] + " " + words[i], 0)
            counts[words[i-1] + " " + words[i]] += 1
            context_counts.setdefault(words[i - 1], 0)
            context_counts[words[i - 1]] += 1
            counts.setdefault(words[i], 0)
            counts[words[i]] += 1
            
#model_file = open("./tutorial02/bigram_model_file.txt", "w")
for ngram, count in counts.items():
        context = ngram.split(" ").pop(0)
        probability = str(float(counts[ngram] / context_counts[context]))
        print(ngram, probability)
        #model_file.write(ngram + " " + probability + "\n")

