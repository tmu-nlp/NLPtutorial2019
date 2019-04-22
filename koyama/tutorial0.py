import sys
from collections import defaultdict

my_dict = defaultdict(lambda: 0)

my_file = open(sys.argv[1], "r")

for line in my_file:
    line = line.strip()
    words = line.split(' ')
    for w in words:
        if w in my_dict:
            my_dict[w] += 1
        else:
            my_dict[w] = 1

for key, value in sorted(my_dict.items()):
    print("%s %d" % (key, value))
