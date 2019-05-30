import sys
from collections import defaultdict

counts = defaultdict(lambda: 0)
total_count = 0

with open('nlptutorial/data/wiki-ja-train.word', 'r') as training_file:
    for line in training_file:
        words = line.strip().split()
        words.append("</s>")
        for word in words:
            counts[word] += 1
            total_count += 1

with open('model-file.txt', 'w') as model_file:
    for key, value in sorted(counts.items()):
        probability = value / total_count
        print(key + ' ' + str(probability), file = model_file)
