from collections import defaultdict, deque, Counter
from tqdm import tqdm
import dill


class Token:
    def __init__(self, id, word, pos, head, attrs):
        self.id = id
        self.word = word
        self.pos = pos
        self.head = head
        self.attrs = attrs
        self.unproc = 0

    def __repr__(self):
        return self.word


def load_data(filename):
    sentences = []
    sentence = []
    for line in open(filename):
        line = line.strip()
        # 1       in      in      IN      IN      _       43      PP
        if line:
            attrs = line.split('\t')
            # ['1', 'in', 'in', 'IN', 'IN', '_', '43', 'PP']
            (rid, rname, rname1, rpos, rpos1, runder, rhead, rdep) = attrs
            token = Token(int(rid), rname, rpos, int(rhead), attrs)
            # class: in
            sentence.append(token)
        elif sentence:
            sentences.append(sentence)
            # yield sentence
            sentence = []

    return sentences


def init_unproc(sentences):
    for sentence in sentences:
        for word in sentence:
            sentence[int(word.head)-1].unproc += 1


def make_feats(stack, queue):
    feat_keys =  []

    # queue: 処理前, stack: 処理中
    if len(queue) > 0 and len(stack) > 0:
        q = queue[0]
        s1 = stack[-1]
        feat_keys += bigram_feat_keys(s1, q, 0)

    if len(stack) > 1:
        s1 = stack[-1]
        s2 = stack[-2]
        feat_keys += bigram_feat_keys(s1, s2, -1)

    feats = Counter(feat_keys)

    return feats


def bigram_feat_keys(prev, next, i):
    return [f'W{i-1}{prev.word}|W{i}{next.word}', f'W{i-1}{prev.word}|P{i}{next.pos}',\
            f'P{i-1}{prev.pos}|W{i}{next.word}',f'P{i-1}{prev.pos}|P{i}{next.pos}']


def shift_reduce(queue, W, mode='train'):
    queue = deque(queue)
    heads = [None] * (len(queue) + 1)
    stack = [Token(0, 'ROOT', 'ROOT', None, None)]

    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)

        score = {}
        for key, weight in W.items():
            for feat, count in feats.items():
                score[key] = weight[feat] * count

        ans = predict_ans(score, len(queue))

        if mode == 'train':
            corr = correct_ans(stack, queue)
            if ans is not corr:
                for feat, count in feats.items():
                    W[ans][feat] -= count
                    W[corr][feat] += count

            # ans = corr

        if ans == 'SHIFT':
            stack.append(queue.popleft())
        elif ans == 'LEFT':
            heads[stack[-2].id] = stack[-1].id
            stack[-1].unproc -= 1
            del stack[-2]
        else:
            heads[stack[-1].id] = stack[-2].id
            stack[-2].unproc -= 1
            del stack[-1]

    return heads


def predict_ans(scores, queue_len):
    ss, sl, sr = scores['SHIFT'], scores['LEFT'], scores['RIGHT']

    if ss >= sl and ss >= sr and queue_len > 0:
        return 'SHIFT'
    elif sl >= sr:
        return 'LEFT'
    else:
        return 'RIGHT'


def correct_ans(stack, queue):
    if len(stack) < 2:
        return 'SHIFT'
    elif stack[-1].head == stack[-2].id and (stack[-1].unproc == 0 or len(queue) == 0):
        return 'RIGHT'
    elif stack[-2].head == stack[-1].id and (stack[-2].unproc == 0 or len(queue) == 0):
        return 'LEFT'
    else:
        return 'SHIFT'


def main():
    W = {}
    W['SHIFT'] = defaultdict(int)
    W['LEFT'] = defaultdict(int)
    W['RIGHT'] = defaultdict(int)
    # *data, = load_data('../../data/mstparser-en-train.dep')
    data = load_data('../../data/mstparser-en-train.dep')
    init_unproc(data)
    for sentence in tqdm(data):
        shift_reduce(sentence, W)

    with open('train_file', 'wb') as f:
        f.write(dill.dumps(W))


if __name__ == '__main__':
    main()

# 41.58223755119638% (1929/4639)
