from collections import defaultdict
import os

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
        for token, count in sorted(tokencounts.items()):
            fout.write(token+'\t'+str(count)+"\n")
            print(token+' '+str(count))
    return out_fname

    # use setdefault

def main():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path=os.path.join(THIS_DIR,os.pardir,os.pardir,'data/wiki-en-train.word')
    CountTokenFreq(input_path,'output.txt')


if __name__ == "__main__":
    main()

