from collections import defaultdict

counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)

with open('../../data/wiki-en-test.word', 'r') as training_file:
    for line in training_file:
        words = line.strip().split()
        words.append("</s>")
        #words.insert(0, "<s>")
        for i in range(1, len(words)):
            counts['\s'.join(words[i - 1:i + 1])] += 1  #bigramの分子
            context_counts[words[i - 1]] += 1          #bigramの分母
            counts[words[i]] += 1                      #unigramの分子
            context_counts[''] += 1
                                #unigramの分母
with open('model-file.txt', 'w') as model_file:
    for key, value in sorted(counts.items()):
        words = key.split('\s')
        words.pop()
        context = ' '.join(words)
        probability = counts[key] / context_counts[context]
        print(key + ' ' + str(probability), file=model_file)
