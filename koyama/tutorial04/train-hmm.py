from collections import defaultdict

emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)

with open('../../data/wiki-en-train.norm_pos', 'r') as input_file:
    for line in input_file:
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

with open('model.txt', 'w') as model_file:
    for key, value in sorted(transition.items()):
        previous, word = key.split(' ')
        print('T ' + key  + ' ' + str(value / context[previous]), file=model_file)
    for key, value in sorted(emit.items()):
        tag, word = key.split(' ')
        print('E ' + key +  ' ' + str(value / context[tag]), file=model_file)
