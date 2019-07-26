import math
from collections import defaultdict
from tqdm import tqdm


def shift_reduce(queue, w):
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1] * (len(queue)+1)
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        score = calc_score(w, feats)
        if (score[0] >= score[1] and score[0] >= score[2] and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif score[1] > score[0] and score[1] > score[2]:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)

    return heads


def load_mst(i_path):
    with open(i_path, 'r') as i_file:
        queue = []
        heads = [-1]
        for line in i_file:
            line = line.rstrip('\n')
            if line:
                id_, word, _, pos, _, _, head, _ = line.split('\t')
                queue.append((int(id_), word, pos))
                heads.append(int(head))
            else:
                yield queue, heads
                queue = []
                heads = [-1]


def shift_reduce_train(queue, heads, w):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        score = calc_score(w, feats)
        if (score[0] >= score[1] and score[0] >= score[2] and len(queue) > 0) or len(stack) < 2:
            ans = 'shift'
        elif score[1] > score[0] and score[1] > score[2]:
            ans = "left"
        else:
            ans = "right"

        if len(stack) < 2:
            corr = "shift"
            stack.append(queue.pop(0))
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            corr = "right"
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            corr = "left"
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            corr = "shift"
            stack.append(queue.pop(0))

        if ans != corr:
            update_weights(w, feats, ans, corr)


def make_features(stack, queue):
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        features[f'W-1 {stack[-1][1]} W-0 {queue[0][1]}'] += 1
        features[f'W-1 {stack[-1][1]} P-0 {queue[0][2]}'] += 1
        features[f'P-1 {stack[-1][2]} W-0 {queue[0][1]}'] += 1
        features[f'P-1 {stack[-1][2]} P-0 {queue[0][2]}'] += 1
    if len(stack) > 1:
        # print(queue)
        features[f'W-2 {stack[-2][1]} W-1 {stack[-1][1]}'] += 1
        features[f'W-2 {stack[-2][1]} P-1 {stack[-1][2]}'] += 1
        features[f'P-2 {stack[-2][2]} W-1 {stack[-1][1]}'] += 1
        features[f'P-2 {stack[-2][2]} P-1 {stack[-1][2]}'] += 1
    return features


def calc_score(w, feats):
    score = [0, 0, 0]
    for key, value in feats.items():
        for i in range(3):
            score[i] += w[i][key] * value
    return score


def update_weights(w, feats, ans, corr):
    for key, value in feats.items():
        if ans == 'shift':
            w[0][key] -= value
        elif ans == 'left':
            w[1][key] -= value
        else:
            w[2][key] -= value

        if corr == 'shift':
            w[0][key] += value
        elif corr == 'left':
            w[1][key] += value
        else:
            w[2][key] += value


def train_sr(train_file, epoch_num=5):
    for _ in tqdm(range(epoch_num)):
        for queue, heads in load_mst(train_file):
            shift_reduce_train(queue, heads, w)


def test_sr(test_file):
    with open('result.txt', 'w') as o_file:
        for queue, _ in load_mst(test_file):
            heads_ = shift_reduce(queue, w)
            heads_.pop(0)
            print(heads_)
            for i in range(len(heads_)):
                result = f'_\t_\t_\t_\t_\t_\t{heads_[i]}'
                print(result, file=o_file)


def main():
    train_path = '../../data/mstparser-en-train.dep'
    test_path = '../../data/mstparser-en-test.dep'
    train_sr(train_path)
    test_sr(test_path)


if __name__ == '__main__':
    w_shift = defaultdict(int)
    w_left = defaultdict(int)
    w_right = defaultdict(int)
    w = [w_shift, w_left, w_right]
    main()
