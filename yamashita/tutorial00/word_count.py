import sys
from collections import defaultdict

counts_dict = defaultdict(lambda: 0)
path = sys.argv[1]
test = sys.argv[2]

with open(path, 'r', encoding='utf-8') as input_file:
    for line in input_file.readlines():
        words = line.strip().split(' ')
        for word in words:
            counts_dict[word] += 1

if test == 'test':
    for key, value in sorted(counts_dict.items()):
        print(key, value)
else:
    print(f'単語の異なり数{len(counts_dict)}')
    print('10単語分の頻度')
    for key, value in sorted(counts_dict.items(), reverse=True)[:10]:
        print(key, value)
