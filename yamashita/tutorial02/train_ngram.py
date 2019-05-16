import sys
from collections import defaultdict

path = sys.argv[1]
N = int(sys.argv[2])
counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)

with open(path, 'r', encoding='utf-8') as i_file:
    for line in i_file.readlines():
        words = line.strip().split(' ')
        words.insert(0, '<s>')
        words.append('</s>')
        for i in range(N-1, len(words)):
            for j in range(1, N):
                counts[' '.join(words[i-N+j:i+1])] += 1
                context_counts[' '.join(words[i-N+j:i])] += 1
            counts[words[i]] += 1
            context_counts[''] += 1

context = []

with open('model_ngram', 'w', encoding='utf-8') as m_file:
    for key, value in sorted(counts.items()):
        words = key.split(' ')
        words.pop(len(words)-1)
        context = ' '.join(words)
        probability = counts[key] / context_counts[context]
        m_file.write(f'{key}\t{probability:.6f}\n')
