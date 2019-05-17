import os
from collections import defaultdict
import sys


print("Hello World")

myDict = defaultdict(lambda: 0)
myDict["alan"] = 22
myDict["bill"] = 45
myDict["eric"] = 33

if "alan" in myDict:
    print("alan exists in myDict")
elif "dab exists in myDict" in myDict:
    print("dab exists in myDict")

print(myDict["eric"])
print(myDict["fred"])

for foo, bar in sorted(myDict.items()):
    print(foo,"-->",bar)

sentence = "this is a pen"
words = sentence.split(" ")

print(" ||| ".join(words))

def aaa(x):
    print(x)

aaa(23)


if len(sys.argv) < 2:
    quit()

myWords = defaultdict(lambda: 0)

myFile = open(sys.argv[1])
for line in myFile:
    line = line.strip()
    for word in line.split(" "):
        myWords[word] += 1

    if len(line) != 0:
        #print(line)

print("number of different words: ", len(myWords))

for key, var in sorted(myWords.items(), reverse=True)[:10]:
    print(key, " occurs ", var , " times")

    