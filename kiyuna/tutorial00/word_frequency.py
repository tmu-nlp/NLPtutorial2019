'''
ファイルの中の単語の頻度を数えるプログラムを作成
'''
import sys
from collections import defaultdict, Counter
from operator import itemgetter as get
import subprocess


def word_frequency(path, is_test=False):
    cnt = defaultdict(int)
    with open(path) as f:
        for line in f:
            for word in line.split():
                cnt[word] += 1          # word.lower()

    print("[+] 単語の異なり数", len(cnt), file=sys.stderr)
    if is_test:
        with open('00-output.txt', 'w') as f:
            for key, value in sorted(cnt.items()):
                print(f'{key}\t{value}', file=f)
        sys.stderr.write("[*] sh check.sh\n")
        subprocess.run(['sh', 'check.sh'])
    else:
        print("数単語の頻度（上位 10 単語のみ）", file=sys.stderr)
        for key, value in sorted(cnt.items(), key=get(1), reverse=True)[:10]:
            print(key, value, file=sys.stderr)


def word_frequency_cnter(path, is_test=False):
    if is_test:
        return
    sys.stderr.write("---------------\n[+] カウンターを使った場合\n")
    with open(path) as f:
        cnt = Counter(f.read().split())
    for key, value in cnt.most_common(10):
        print(key, value, file=sys.stderr)


if __name__ == '__main__':
    '''
    「python word* test」-> テスト
    '''
    TEST = sys.argv[1:] == ["test"]
    path = '../../test/00-input.txt' if TEST else '../../data/wiki-en-train.word'
    word_frequency(path, TEST)

    word_frequency_cnter(path, TEST)
