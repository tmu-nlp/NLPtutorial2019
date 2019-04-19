import sys
from collections import defaultdict

counts_dict = defaultdict(lambda:0)
path = sys.argv[1]

with open(path,'r',encoding='utf-8') as input_file:
    for line in input_file.readlines():
        words = line.strip().split(' ')
        for word in words:
            counts_dict[word] += 1


for key,value in sorted(counts_dict.items()):
    print('{}:{}'.format(key,value))