import math
from collections import defaultdict


def write_model(train_path: str, model_path: str):
    with open(train_path, "r") as f:
        total_count = 0
        counts = defaultdict(lambda: 0)
        for line in f:
            words = line.rstrip().split()
            for word in words:
                if counts[word] != 0:
                    counts[word] += 1
                    total_count += 1
                else:
                    counts[word] = 1
                    total_count += 1
    with open(model_path, "w") as f:
        for k, v in counts.items():
            p = float(v) / total_count
            f.write(f"{k}\t{p:f}\n")


def load_model(model_path: str) -> {}:
    with open(model_path, "r") as f:
        p = defaultdict(lambda: 0)
        for line in f:
            w_and_p = line.rstrip().split()
            p[w_and_p[0]] = float(w_and_p[1])
    return p


def Viterbi_algorithm(uni_prob: {}, test_path: str, test_ans_path: str):
    with open(test_path, "r") as f, open(test_ans_path, "w") as fp:
        best_edge = defaultdict(lambda: 0)
        best_score = defaultdict(lambda: 0)
        lam_1 = 0.95
        lam_unk = 1-lam_1
        N = 100000
        for line in f:
            line = line.rstrip().split()  # 形態素解析済の日本語
            # line = list(line.rstrip())  # 英語
            best_edge = [None] * (len(line)+1)
            best_score = [0] * (len(line)+1)
            for word_end in range(1, len(line) + 1):
                best_score[word_end] = math.pow(10, 10)
                for word_begin in range(word_end):
                    word = ""
                    for i in line[word_begin:word_end]:
                        word = word + i
                    if not(uni_prob[word] != 0 or len(word) == 1):
                        continue
                    prob = lam_1*uni_prob[word] + lam_unk/N
                    my_score = best_score[word_begin] - math.log(prob)
                    if not(my_score < best_score[word_end]):
                        continue
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
            words = []
            next_edge = best_edge[-1]
            while next_edge != None:
                word = line[next_edge[0]:next_edge[1]]
                words.append("".join(word))
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            fp.write("  ".join(words) + "\n")


if __name__ == "__main__":
    #テスト
    # model_path = "04-model.txt"
    # test_path = "04-input.txt"
    # test_ans_path = "04-output.txt"
    #本番
    train_path = "wiki-ja-train.word"
    model_path = "model.txt"
    test_path = "wiki-ja-test.word"
    test_ans_path = "wiki-ja-test-ans.txt"
    write_model(train_path, model_path)
    uni_prob = load_model(model_path)
    Viterbi_algorithm(uni_prob, test_path, test_ans_path)