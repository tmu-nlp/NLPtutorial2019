import sys
from collections import defaultdict

file_path = sys.argv[1]
word_count_dict = defaultdict(lambda: 0)
total = 0

with open(file_path, 'r', encoding='utf-8') as i_file:
    for line in i_file.readlines():
        words = line.strip().split(' ')
        words.append('</s>')
        for word in words:
            word_count_dict[word] += 1
            total += 1

with open('model', 'w', encoding='utf-8') as m_file:
    for key, value in sorted(word_count_dict.items()):
        probability = word_count_dict[key] / total
        m_file.write(f'{key} {probability:.6f}\n')
