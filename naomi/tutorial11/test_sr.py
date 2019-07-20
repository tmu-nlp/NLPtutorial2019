from collections import deque
from train_sr import TOK, ACT, import_CoNLL, classify_action, dd, make_feats
import pickle
from copy import copy


def shift_reduce(ws, wl, wr, queue: deque):
    '''
    入力: 素性に対する重みとqueue
    ws, wl, wr: shift, reduce left, reduce right の重み
    queue = [(1, word1, POS1), (2, word2, POS2), ...]

    戻り値: 各単語の親のIDを格納した配列
    heads = [-1, head1, head2, ...]
    '''

    # はじめStackにはROOTだけ存在
    stack = [TOK(0, 'ROOT', 'ROOT')]

    # 親のIDが入る
    heads = [0]*len(queue)

    # while len(queue) > 0 or len(stack) > 0:
    while len(queue) > 0:

        phi = make_feats(stack, queue)

        action = classify_action(ws, wl, wr, phi)

        if action == ACT.L and len(stack) >= 2:
            heads[stack[-2].id] = stack[-1].id
            stack.pop(-2)
        elif action == ACT.R and len(stack) >= 2:
            heads[stack[-1].id] = stack[-2].id
            stack.pop(-1)
        else:
            stack.append(queue.popleft())

    return heads


if __name__ == "__main__":
    with open(f"./pickles/weight", 'rb') as f_in:
        (ws, wl, wr) = pickle.load(f_in)

    ques_heads_list = import_CoNLL('../../data/mstparser-en-test.dep')

    with open('result.txt', 'w+', encoding='utf-8') as f_out:
        for (queue, _) in ques_heads_list:
            heads = shift_reduce(ws, wl, wr, copy(queue))
            for i, tok in enumerate(queue):
                print(f'{tok.id}\tsurface\t{tok.word}\t{tok.pos}\tpos2\text\t{heads[i]}\tlabel', file=f_out)
            print('', file=f_out)