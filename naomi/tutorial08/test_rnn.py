import numpy as np

class TestRNN():
    def __init__(self, net, word_ids, pos_ids):
        self.net = net
        self.word_ids = word_ids
        self.pos_ids = pos_ids

        self.vocab_size = len(self.word_ids)
        self.pos_size = len(self.pos_ids)

    def create_onehot(self, id, size):
        v = np.zeros(size)
        v[id] = 1
        return v
    
    def softmax(self, vec):
        const = max(vec)
        exp_vec = np.exp(vec-const)
        sum_vec = np.sum(exp_vec)
        return exp_vec / sum_vec
    
    def forward(self, words):
        # words : 文の onehot ベクトル
        # 単語数
        vocab = len(words)
        # 時刻 t における各値
        h = [np.ndarray for _ in range(vocab)]  # 隠れ層の値 h[t]
        p = [np.ndarray for _ in range(vocab)]  # 出力の確率分布 p[t]
        y = [0 for _ in range(vocab)]

        w_rx, w_rh, w_oh, b_r, b_o = self.net
        for t in range(vocab):
            if t != 0:
                # 入力,前の時刻の両方の重みで現時刻の隠れ層の値を求める
                current = np.dot(w_rh, h[t - 1]) + b_r
                prev = np.dot(w_rx, words[t])
                h[t] = np.tanh(current + prev)
            else:
                # t = 0 のとき、前の時刻の隠れ層を考慮しない
                current = np.dot(w_rx, words[t]) + b_r
                h[t] = np.tanh(current)
            # 各時刻の出力
            p[t] = self.softmax(np.dot(w_oh, h[t]) + b_o)
            # 各時刻の出力の最大値のインデックス (tuple(ndarray,) で返すので [0][0] を指定する必要がある)
            y[t] = np.where(p[t] == max(p[t]))[0][0]

        return h, p, y

    def test(self, path):

        for line in open(path, "r", encoding="utf-8"):
            words = line.strip().split()
            ws = [None for _ in range(len(words))]
            for i, word in enumerate(words):
                if word in self.word_ids:
                    ws[i] = self.create_onehot(self.word_ids[word], self.vocab_size)
                else:
                    ws[i] = np.zeros(self.vocab_size)
            h, p, y = self.forward(ws)
        
            for tag in y:
                for key, val in self.pos_ids.items():
                    if val == tag:
                        print(key, end=" ")
            print()
        

if __name__ == "__main__":
    main()

"""
epoch = 3, node = 20
Accuracy: 79.95% (3648/4563)
Most common mistakes:
JJ --> NN       148
NNS --> NN      134
NNP --> NN      71
RB --> NN       55
VBG --> NN      38
VBN --> NN      33
CD --> NN       29
VBN --> JJ      28
VBP --> NN      21
IN --> WDT      16
"""


