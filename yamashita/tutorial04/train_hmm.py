import sys
from collections import defaultdict

emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)
i_path = sys.argv[1]

with open(i_path, 'r', encoding='utf-8') as i_file:
    for line in i_file:
        previous = '<s>'
        context[previous] += 1
        wordtags = line.rstrip().split(' ')
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            transition[previous + ' ' + tag] += 1
            context[tag] += 1
            emit[tag + ' ' + word] += 1
            previous = tag
        transition[previous + ' </s>'] += 1


for key, value in sorted(transition.items()):
    previous, word = key.split(' ')
    print(f'T {key} {value / context[previous]:.6f}')
for key, value in sorted(emit.items()):
    tag, word = key.split(' ')
    print(f'E {key} {value / context[tag]:.6f}')
