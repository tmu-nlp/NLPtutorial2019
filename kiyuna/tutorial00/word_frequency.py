'''
ファイルの中の単語の頻度を数えるプログラムを作成
'''
from collections import defaultdict, Counter


def count_word_freq(path, trans=str):
    """ Word Frequency Counter

    指定されたファイルについて，単語の分布を返す．

    Args:
        path: 対象とするファイルのパス
        trans: 単語を
            大文字と小文字の区別をなくしたいとき ->「lambda x: x.lower()」

    Returns:
        defaultdict: {単語: 出現数}

    """
    cnt = defaultdict(int)
    with open(path) as f:
        for line in f:
            for word in map(trans, line.split()):
                cnt[word] += 1
    return cnt


def word_frequency_cnter(path):
    with open(path) as f:
        cnt = Counter(f.read().split())
    return cnt


if __name__ == '__main__':
    '''
    「python word* test」-> テスト
    '''
    import os
    import sys
    import subprocess
    from operator import itemgetter as get

    os.chdir(os.path.dirname(__file__))     # cd .

    def message(text):
        print("\33[92m" + text + "\33[0m")

    is_test = sys.argv[1:] == ["test"]

    message("[*] count word frequency")
    if is_test:
        path = '../../test/00-input.txt'
    else:
        path = '../../data/wiki-en-train.word'
    cnt = count_word_freq(path)

    if is_test:
        fn_out = '00-out.txt'
        with open(fn_out, 'w') as f:
            for key, value in sorted(cnt.items()):
                print(f'{key}\t{value}', file=f)

        message("[*] sh check.sh")
        # 'test/00-answer.txt' と比較
        subprocess.run(f'diff -s {fn_out} ../../test/00-answer.txt'.split())
        os.remove(fn_out)

    else:
        print("[+] 単語の異なり数:", len(cnt), "タイプ")
        print("[+] 数単語の頻度（上位 10 単語のみ）")
        for key, value in sorted(cnt.items(), key=get(1), reverse=True)[:10]:
            print(key, value)

        message("[*] collections.Counter を使った場合")
        cnt = word_frequency_cnter(path)
        for key, value in cnt.most_common(10):
            print(key, value, file=sys.stderr)

        message("[*] trans=lambda x: x.lower() と指定した場合")
        cnt = count_word_freq(path, trans=lambda x: x.lower())
        for key, value in sorted(cnt.items(), key=get(1), reverse=True)[:10]:
            print(key, value)

    message("[+] Finished!")
