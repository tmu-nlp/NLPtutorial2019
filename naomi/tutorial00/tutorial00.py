from collections import defaultdict

def CountTokenFreq(in_fname,out_fname):

    tokencounts = defaultdict(lambda:0)

    with open(in_fname) as fin, open(out_fname,'w') as fout:
        for line in fin:
            words = line.split()
            for word in words:
                if word in tokencounts:
                    tokencounts[word]+=1
                else:
                    tokencounts[word]=1
        for token, count in tokencounts.items():
            fout.write(token+'\t'+str(count)+"\n")
            print(token+' '+str(count))
    return out_fname

