import os


def trainunigram(infile: str, outfile: str):
    with open(infile,encoding='utf-8') as fin, open(outfile, 'w+', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        wcounts = {}

        for line in lines:
            words = line.replace("\n"," </s>").split(" ")
            # words = line.split(" ")
            for word in words:
                # word.replace("\n","<s>")
                wcounts.setdefault(word, 0)
                wcounts[word] += 1

        for token, count in sorted(wcounts.items()):          
            unigram = count/sum(wcounts.values())

            print(token+' {0:.6f}'.format(unigram))
            fout.write(token+'\t{0:.6f}\n'.format(unigram))

    return

def testunigram(infile: str, outfile:str):
    with open(infile,encoding='utf-8') as fin, open(outfile, 'w+', encoding='utf-8') as fout:
        lines=fin.readlines()

        fout.write('test')
    return

def main():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path=os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-input.txt')
    trainunigram(input_path,'output.txt')

if __name__ == "__main__":
    main()
