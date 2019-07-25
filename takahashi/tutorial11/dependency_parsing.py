from collections import deque
from collections import defaultdict as dd
from tqdm import tqdm

w = {"SHIFT": dd(float), "LEFT": dd(float), "RIGHT": dd(float)}

def file_parse(file_path):
    sentence = []
    for line in open(file_path, "r", encoding="utf-8"):
        line = line.strip()

        if len(line) > 0:
            id, word, _, pos, _, _, head, _ = line.split("\t")
            sentence.append({
                "id": int(id), 
                "word": word, 
                "pos": pos, 
                "head": int(head),
                "unproc": 0
            })

        elif sentence:
            yield sentence
            sentence = []

def preprocess(queue):
    for token in queue:
        if token["head"] != 0:
            continue

        # 各単語の未処理の child の数 unproc をカウント
        parent = token["head"] - 1
        queue[parent]["unproc"] += 1

def make_feats(stack, queue):
    feats = {}

    if len(queue) > 0 and len(stack) > 0:
        s = stack[-1]
        q = queue[0]
        feats[f"W-1 {s['word']} W-0 {q['word']}"] = 1
        feats[f"W-1 {s['word']} W-0 {q['pos']}"] = 1
        feats[f"W-1 {s['pos']} W-0 {q['word']}"] = 1
        feats[f"W-1 {s['pos']} W-0 {q['pos']}"] = 1

    if len(stack) > 1:
        s1, s2 = stack[-1], stack[-2]
        feats[f"W-1 {s2['word']} W-0 {s1['word']}"] = 1
        feats[f"W-1 {s2['word']} W-0 {s1['pos']}"] = 1
        feats[f"W-1 {s2['pos']} W-0 {s1['word']}"] = 1
        feats[f"W-1 {s2['pos']} W-0 {s1['pos']}"] = 1

    return feats

def calculate_ans(feats, queue):
    ss, sl, sr = 0, 0, 0
    action = "RIGHT"
    for key, val in feats.items():
        ss += w["SHIFT"][key] * val
        sl += w["LEFT"][key] * val
        sr += w["RIGHT"][key] * val

    if ss >= sl and ss >= sr and len(queue) > 0:
        action = "SHIFT"
    elif sl >= sr:
        action = "LEFT"

    return action

def calculate_cor(stack, queue):
    action = "LEFT"

    if not queue:
        if stack[-1]["head"] == stack[-2]["id"]:
            action = "RIGHT"
        return action

    if stack[-1]["head"] == stack[-2]["id"] and stack[-1]["unproc"] == 0:
        action = "RIGHT"
    elif stack[-2]["head"] == stack[-1]["id"] and stack[-2]["unproc"] == 0:
        action = "LEFT"
    else:
        action = "SHIFT"

    return action

def update_weights(feats, ans, cor):
    for key, val in feats.items():
        w[ans][key] -= val
        w[cor][key] += val

def shift_reduce_train(queue):
    stack = [{"id": 0, "word": "ROOT", "pos": "ROOT", "head": None, "unproc": -1}]

    while len(queue) > 0 or len(stack) > 1:
        if len(stack) <= 1:
            stack.append(queue.popleft())
            continue

        feats = make_feats(stack, queue)
        ans = calculate_ans(feats, queue)
        cor = calculate_cor(stack, queue)

        if ans != cor:
            update_weights(feats, ans, cor)

        if cor == "SHIFT":
            stack.append(queue.popleft())
        elif cor == "LEFT":
            stack[-1]["unproc"] -= 1
            del stack[-2]
        else:
            stack[-2]["unproc"] -= 1
            del stack[-1]

def shift_reduce(queue):
    # タブ区切りで rnum, rname, rname1, rpos, rpos1, runder, rhead, rdep の出力形式
    # ただし評価において rhead しか見ていないので rhead 以外 _ で出力
    preds = ["_\t" * 6 + "0\t_"] * len(queue)
    stack = [{"id": 0, "word": "ROOT", "pos": "ROOT", "head": None, "unproc": -1}]

    while len(queue) > 0 or len(stack) > 1:
        if len(stack) <= 1:
            stack.append(queue.popleft())
            continue

        feats = make_feats(stack, queue)
        ans = calculate_ans(feats, queue)

        if ans == "SHIFT":
            stack.append(queue.popleft())
        elif ans == "LEFT":
            preds[stack[-2]["id"] - 1] = "_\t" * 6 + f"{stack[-1]['id']}\t_"
            del stack[-2]
        else:
            preds[stack[-1]["id"] - 1] = "_\t" * 6 + f"{stack[-2]['id']}\t_"
            del stack[-1]

    return preds


if __name__ == "__main__":
    train_path = "../files/data/mstparser-en-train.dep"
    test_path = "../files/data/mstparser-en-test.dep"
    epochs = 10

    # train
    for _ in tqdm(range(epochs), desc="epoch"):
        for sentence in file_parse(train_path):
            queue = deque(sentence)
            preprocess(queue)
            shift_reduce_train(queue)

    # test
    with open("ans", "w", encoding="utf-8") as f:
        for sentence in file_parse(test_path):
            queue = deque(sentence)
            preds = shift_reduce(queue)
            print("\n".join(preds) + "\n", file=f)