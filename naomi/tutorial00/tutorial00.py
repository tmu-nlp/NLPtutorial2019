from collections import defaultdict
import os


def CountTokenFreq(in_fname, out_fname):

    # tokenを納める辞書を初期化
    tokencounts = defaultdict(lambda: 0)

    with open(in_fname) as fin, open(out_fname, 'w') as fout:
        for line in fin:
            # 1行を単語ごとに分ける
            words = line.split()
            # 単語について辞書に格納
            for word in words:
                tokencounts[word] += 1
        
        for token, count in sorted(tokencounts.items()):
            fout.write(token+'\t'+str(count)+"\n")
            print(token+' '+str(count))
    return out_fname

    # use setdefault


def main():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    path = 'data/wiki-en-train.word'
    input_path = os.path.join(THIS_DIR, os.pardir, os.pardir, path)
    CountTokenFreq(input_path, 'output.txt')


if __name__ == "__main__":
    main()
