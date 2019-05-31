train_file =  open("/Users/hongfeiwang/desktop/nlptutorial-master/test/05-train-input.txt","r").readlines()
emit = {}
transition = {}
context = {}
for line in train_file:
    previous = "<s>"
    context.setdefault(previous, 0)
    context[previous] += 1
    for wordtag in line.strip().split(" "):
        word, tag = wordtag.split("_")
        transition.setdefault(previous+" "+tag, 0)
        transition[previous+" "+tag] += 1
        context.setdefault(tag, 0)
        context[tag] += 1
        emit.setdefault(tag+" "+word, 0)
        emit[tag+" "+word] += 1
        previous = tag
    transition.setdefault(previous+" </s>",0)
    transition[previous+" </s>"] += 1
model_file = open("./tutorial04/model_file.txt","w")
for key, value in transition.items():
    previous, word = key.split(" ")
    model_file.write("T"+" "+key+" "+str(float(value/context[previous]))+"\n")
for key, value in emit.items():
    previous, word = key.split(" ")
    model_file.write("E"+" "+key+" "+str(float(value/context[previous]))+"\n")