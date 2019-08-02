from train_rnn import *
from test_rnn import *
import subprocess


def main():
    train_path = '../../data/wiki-en-train.norm_pos'
    train_rnn(train_path, node_num=5, epoch_num=10, λ=0.01)

    test_path = '../../data/wiki-en-test.norm'
    out_path = './out.txt'
    test_rnn(test_path, out_path)

    script_path = '../../script/gradepos.pl'
    ans_path = '../../data/wiki-en-test.pos'
    subprocess.run(f'{script_path} {ans_path} {out_path}'.split())


def test():
    train_path = '../../test/05-train-input.txt'
    net, x_ids, y_ids = train_rnn(train_path, node_num=5, epoch_num=10, λ=0.01)

    test_path = '../../test/05-test-input.txt'
    out_path = './out.txt'
    test_rnn(test_path, out_path)

    script_path = '../../script/gradepos.pl'
    ans_path = '../../test/05-test-answer.txt'
    subprocess.run(f'{script_path} {ans_path} {out_path}'.split())


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['test']:
        message('test')
        test()
    else:
        message('main')
        main()


''' RESULT
**test こまった**
Accuracy: 50.00% (3/6)


Most common mistakes:
Y --> X 2
Z --> X 1

**main**
Accuracy: 80.89% (3691/4563)

Most common mistakes:
NN --> NNP      97
JJ --> NNP      47
NNS --> NNP     47
-RRB- --> NNP   43
-LRB- --> NNP   41
VBN --> JJ      34
VBN --> NNP     30
RB --> NNP      28
CD --> VBG      25
NNP --> NN      24
'''
