from itertools import product
from math import log2
import sys
import subprocess
from collections import defaultdict


def message(text="", CR=False):
    text = "\r" + text if CR else text + "\n"
    sys.stdout.write("\33[92m" + text + "\33[0m")


def train_hmm(input_path, λ_emit=0.95, λ_trans=0.95):
    """ #5 p8 """
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)

    possible_tags = {'<s>', '</s>'}
    N = 0
    for line in map(lambda x: x.rstrip(), open(input_path)):
        previous = "<s>"    # 文頭記号
        context[previous] += 1
        wordtags = line.split(" ")
        for wordtag in wordtags:
            N += 1
            word, tag = wordtag.split('_')
            transition[f'{previous} {tag}'] += 1    # 遷移を数え上げる
            context[tag] += 1                       # 文脈を数え上げる
            emit[f'{tag} {word}'] += 1              # 生成を数え上げる
            previous = tag
            possible_tags.add(tag)
        transition[f'{previous} </s>'] += 1
    # 遷移確率を出力
    # N = 10000
    # print(f'N = {N}')
    corr_trans = (1 - λ_trans) / N
    corr_emit = (1 - λ_emit) / N
    transition_prob = defaultdict(lambda: corr_trans)
    emit_prob = defaultdict(lambda: corr_emit)
    for key, value in transition.items():
        previous, _ = key.split(' ')
        transition_prob[key] = λ_trans * value / context[previous] + corr_trans
    # 同じく生成確率を出力 (「T」ではなく「E」を付与)
    for key, value in emit.items():
        tag, _ = key.split(' ')
        emit_prob[key] = λ_emit * value / context[tag] + corr_emit
    return transition_prob, emit_prob, possible_tags


def test_hmm(test_path, out_path, transition, emit, possible_tags, beam_size=4):
    """ #5 p19 """

    def update_score(best_score, best_edge, score, prv, nxt):
        if nxt not in best_score or best_score[nxt] > score:
            best_score[nxt] = score
            best_edge[nxt] = prv

    with open(out_path, 'w') as f_out:
        for j, words in enumerate(map(lambda x: x.rstrip().split(' '), open(test_path))):
            # Foward Step
            l = len(words)
            best_score = {'0 <s>': 0}   # <s> から始まる
            best_edge = {'0 <s>': None}
            activate_tags = ['<s>']     # new
            for i in range(l):
                # print(i, end=" ")
                my_best = {}            # new
                for prv, nxt in product(activate_tags, possible_tags):   # new
                    i_prv = f'{i} {prv}'
                    prv_nxt = f'{prv} {nxt}'
                    if i_prv not in best_score:
                        continue
                    score = best_score[i_prv] - log2(transition[prv_nxt]) \
                        - log2(emit[f'{nxt} {words[i]}'])
                    update_score(best_score, best_edge, score, i_prv, f'{i + 1} {nxt}')
                    my_best[nxt] = score
                my_best_sorted = sorted(my_best.items(), key=lambda x: x[1])
                activate_tags = [key for key, value in my_best_sorted[:beam_size]]
            for tag in activate_tags:
                my_best = {}
                l_tag = f'{l} {tag}'
                tag_eos = f'{tag} </s>'
                if l_tag not in best_score:
                    continue
                score = best_score[l_tag] - log2(transition[tag_eos])
                update_score(best_score, best_edge, score, l_tag, f'{l + 1} </s>')
            tags = []
            nxt_edge = best_edge[f'{l + 1} </s>']
            while nxt_edge != '0 <s>':
                # このエッジの品詞を出力に追加
                position, tag = nxt_edge.split(' ')
                tags.append(tag)
                nxt_edge = best_edge[nxt_edge]
            tags.reverse()
            print(' '.join(tags), file=f_out)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        message("[*] test")
        train_path = '../../../nlptutorial/test/05-train-input.txt'
        test_path = '../../../nlptutorial/test/05-test-input.txt'
        ans_path = '../../../nlptutorial/test/05-test-answer.txt'
    else:
        message("[*] main")
        train_path = '../../../nlptutorial/data/wiki-en-train.norm_pos'
        test_path = '../../../nlptutorial/data/wiki-en-test.norm'
        ans_path = '../../../nlptutorial/data/wiki-en-test.pos'
    out_path = 'out.txt'

    transition, emit, possible_tags = train_hmm(train_path, 0.95)
    test_hmm(test_path, out_path, transition, emit, possible_tags, 2)

    script_path = '../../../nlptutorial/script/gradepos.pl'
    subprocess.run(f'perl {script_path} {ans_path} {out_path}'.split())
