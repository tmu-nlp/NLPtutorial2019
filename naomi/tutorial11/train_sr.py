from collections import defaultdict
from collections import deque
from enum import Enum
import pickle
from tqdm import tqdm
from copy import copy


class ACT(Enum):
    S = 'Do Shift'
    L = 'Reduce Left'
    R = 'Reduce Right'


class Token:
    def __init__(self, i, w, p, h):
        self.id = i
        self.word = w
        self.pos = p
        self.head = h
        self.unproc = 0


def import_CoNLL(path):
    '''
    入力；CoNLLファイルのpath
    出力：ques_heads_list = [(queue1, heads1), (queue2, heads2), ...]

    CoNLLファイル
        ID 単語 原型 品詞 品詞 2 拡張 親 ラベル
        1 ms. ms. NNP NNP _ 2 DEP
        2 haag haag NNP NNP _ 3 NP-SBJ
        3 plays plays VBZ VBZ _ 0 ROOT
        4 elianti elianti NNP NNP _ 3 NP-OBJ
        5 . . . . _

    '''
    ques_list = []
    queue = deque([])
    procs = defaultdict(int)

    for line in open(path, 'r', encoding='utf-8'):
        if line != '\n':
            i, surface, word, pos, pos2, _, head, label = line.strip().split()
            queue.append(Token(int(i), word, pos, int(head)))
            procs[int(head)] += 1
        else:
            for j, tok in enumerate(queue):
                if tok.id in procs:
                    queue[j].unproc = procs[tok.id]
            ques_list.append(queue)
            queue = deque([])
            procs = defaultdict(int)
    
    return ques_list


def shift_reduce_train(queue: deque, ws, wl, wr):
    '''
    入力:
    queue = [tok1, tok2, ...]

    出力: 各素性の，shift, reduce left, reduce rightに対する重み
    ws, wl, wr
    '''

    stack = [Token(0, 'ROOT', 'ROOT', 0)]

    while len(queue) > 0 or len(stack) > 1:

        # stackに２つなければshift
        if len(stack) < 2:
            stack.append(queue.popleft())
            continue

        # stack, queueの単語と品詞を元に素性を計算
        phi = make_feats(stack, queue)

        # アクションを分類してみる
        pred_action = classify_action(ws, wl, wr, phi, queue)

        # 正しいアクションを求める
        corr_action = calc_corr(stack, queue)

        # 間違った答えを返したら重みを更新する
        if pred_action != corr_action:
            for name, v in phi.items():
                value = copy(v)
                if pred_action == ACT.S:
                    ws[name] -= value
                elif pred_action == ACT.L:
                    wl[name] -= value
                elif pred_action == ACT.R:
                    wr[name] -= value

                if corr_action == ACT.S:
                    ws[name] += value
                elif corr_action == ACT.L:
                    wl[name] += value
                elif corr_action == ACT.R:
                    wr[name] += value

        # action according to corr
        ans_action = classify_action(ws, wl, wr, phi, queue)

        if ans_action == ACT.S:
            stack.append(queue.popleft())

        elif ans_action == ACT.L:
            stack[-1].unproc -= 1
            stack.pop(-2)
        elif ans_action == ACT.R:
            stack[-2].unproc -= 1
            stack.pop(-1)


def classify_action(ws, wl, wr, phi, queue):
    '''
    Classify action to be taken
    '''

    # スコアの初期化
    ss, sl, sr = 0, 0, 0

    # shift, reduce left, reduce rightのスコアを計算
    for name, value in phi.items():
        ss += ws[name] * value
        sl += wl[name] * value
        sr += wr[name] * value
    
    if ss >= sl and ss >= sr and queue:
        return ACT.S
    elif sl >= sr:
        return ACT.L
    else:
        return ACT.R


def calc_corr(stack, queue):
    '''
    各単語の未処理の子供の数unprocを考慮して「正解」の計算をする
    '''
    # queueに１つもなければ，Reduce
    if not queue:
        # 左が右の親のとき
        if stack[-1].head == stack[-2].id:
            return ACT.R
        else:
            return ACT.L

    # 左が右の親のとき
    if len(stack) >= 2 and stack[-1].head == stack[-2].id \
       and stack[-1].unproc == 0:
        return ACT.R
    
    # 右が左の親のとき
    elif len(stack) >= 2 and stack[-2].head == stack[-1].id \
        and stack[-2].unproc == 0:
        return ACT.L
    else:
        return ACT.S


def make_feats(stack, queue):
    '''
    入力:
    stack = [..., (2, saw, VBD), (3, a, DET)]
    queue = [(4, girl, pos4), (5, word5, pos5), ...]

    出力: 素性
    phi = {w-2_saw_w-1_a: 1, w-2_saw_p-1_DET, ...}
    '''

    phi = defaultdict(float)
    right = stack[-1]

    if len(stack) >= 2:
        left = stack[-2]
        phi[f'W-2{left.word},W-1{right.word}'] += 1
        phi[f'W-2{left.word},P-1{right.pos}'] += 1

        phi[f'P-2{left.pos},W-1{right.word}'] += 1
        phi[f'P-2{left.pos},P-1{right.pos}'] += 1

    if stack and queue:
        phi[f'W-1{stack[-1].word},W0{queue[0].word}'] += 1
        phi[f'W-1{stack[-1].word},P0{queue[0].pos}'] += 1

        phi[f'P-1{stack[-1].pos},W0{queue[0].word}'] += 1
        phi[f'P-1{stack[-1].pos},P0{queue[0].pos}'] += 1

    return phi


if __name__ == "__main__":
    # CoNLLファイルの読み込み
    ques_list = import_CoNLL('../../data/mstparser-en-train.dep')

    epoch = 20

    # 重みの初期化
    ws = defaultdict(float)
    wl = defaultdict(float)
    wr = defaultdict(float)

    for _ in tqdm(range(epoch)):
        for queue in ques_list:
            shift_reduce_train(queue, ws, wl, wr)

    with open('./pickles/weight', 'wb') as f_out:
        pickle.dump((ws, wl, wr), f_out)
