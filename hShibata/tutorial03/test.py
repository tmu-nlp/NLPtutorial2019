
print("hello")

# the test to change value of dictionary in for loop by smart way. We need this test to implement a method which builds n-gram using a recursive function.
myDict = {}

myDict[0] = 3
myDict[1] = 3

for key, val in myDict.items():
    val = 0

print(myDict) # output is {0: 3, 1: 3}. So we cannot.

for key, val in myDict.items():
    myDict[key] = 99

print(myDict) # output is {0: {2: 2}, 1: {2: 2}}. So we can by this way, which re reference using key, as we know well.

class tClass:
    def __init__(self):
        self.a = 0


# next we will shows to change the member value of a class using for loop.
for key, val in myDict.items():
    myDict[key] = tClass() # first, initialize all value as a class.

for key, val in myDict.items():
    val.a = 99 # then we can change its member, while we could not for an ordinal value type such as number.

for key, val in myDict.items():
    print(key, val.a) 
# output is 0 99\n 1 99. So this is possible. We can know that python3 is implemented to pass a reference of a dictionary value on the for loop, if it is a class.

# the below example test whether the next syntax is recognized correctly.
def call(ins: tClass): 
    print(ins.a)

for key, val in myDict.items():
    call(val)
# the output is 99\n 99. So this works.