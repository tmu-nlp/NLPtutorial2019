from collections import defaultdict


def train_hmm(path_r, path_w):
    with open(path_r, "r") as fr, open(path_w, "w+") as fw:
        emit = defaultdict(lambda: 0)
        transition = defaultdict(lambda: 0)
        context = defaultdict(lambda: 0)
        for line in fr:
            previous = "<s>"  # 文頭記号
            context[previous] += 1
            wordtags = line.rstrip().split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.rstrip().split("_")
                transition[previous+" "+tag] += 1  # 遷移を数え上げる
                context[tag] += 1  # 文脈を数え上げる
                emit[tag+" "+word] += 1  # 生成を数え上げる
                previous = tag
            transition[previous+" "+"</s>"] += 1
        # 遷移確率を出力
        for key, value in sorted(transition.items()):
            previous, word = key.split(" ")
            print(f"T {key} {value/context[previous]:.6f}", file=fw, )
        for key, value in sorted(emit.items()):
            previous, word = key.split(" ")
            print(f"E {key} {value/context[previous]:.6f}", file=fw, )
        

if __name__ == "__main__":
    # path_r = "05-train-input.txt"
    # path_w = "05-train-output.txt"
    path_r = "wiki-en-train.norm_pos"
    path_w = "wiki-model.txt"
    train_hmm(path_r, path_w)
