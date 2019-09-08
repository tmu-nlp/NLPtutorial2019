import random
from math import log
from collections import defaultdict
from tqdm import tqdm
from sys import argv


random.seed(42)


class Topic_Model:
    def __init__(self, iteration_num, NUM_TOPICS, α, β, is_tqdm=True):
        self.iteration_num = iteration_num
        self.NUM_TOPICS = NUM_TOPICS
        self.α = α
        self.β = β
        self.is_tqdm = is_tqdm
        self.num_wordtype = 0
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)

    def sampleone(self, probs):
        z = sum(probs)  # 確率の和(正規化項 ) を計算
        remaining = random.random() * z
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
        print('error')
        exit()

    def addCounts(self, word, topic, docid, amount):
        assert topic >= 0
        assert docid >= 0
        assert amount >= 0 or amount == -1

        self.xcounts[topic] += amount
        self.xcounts[f'{word}|{topic}'] += amount

        self.ycounts[docid] += amount
        self.ycounts[f'{topic}|{docid}'] += amount
        # バグチェック
        # < 0 の場合はエラー終了

    def init(self, in_path):
        wordtype = []
        for line in map(lambda x: x.rstrip(), open(in_path)):
            docid = len(self.xcorpus)
            words = line.split()
            topics = []
            for word in words:
                topic = random.randint(0, self.NUM_TOPICS - 1)
                topics.append(topic)
                self.addCounts(word, topic, docid, 1)  # カウントを追加
                if word not in wordtype:
                    wordtype.append(word)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)
        self.num_wordtype = len(wordtype)

    def sampling(self, out_path='out.txt'):
        for _ in tqdm(range(self.iteration_num)):
            ll = 0
            for i in tqdm(range(len(self.xcorpus))):
                for j in range(len(self.xcorpus[i])):
                    probs = []
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.addCounts(x, y, i, -1)  # 各種カウントの減算 (-1)
                    for k in range(self.NUM_TOPICS):
                        p_x_k = (self.xcounts[f'{x}|{k}'] + self.α) / \
                            (self.xcounts[k] + self.α * self.num_wordtype)
                        p_k_y = (self.ycounts[f'{k}|{i}'] + self.β) / \
                            (self.ycounts[i] + self.β * self.NUM_TOPICS)
                        probs.append(p_x_k * p_k_y)      # トピック k の確率
                    new_y = self.sampleone(probs)
                    ll += log(probs[new_y])        # 対数尤度の計算
                    self.addCounts(x, new_y, i, 1)      # 各カウントの加算
                    self.ycorpus[i][j] = new_y
        #         print(ll)
        # print(self.xcounts)
        # print(self.ycounts)
        # print(self.xcorpus)
        # print(self.ycorpus)

        # word_topic = {i: [] for i in range(self.NUM_TOPICS)}
        word_topic_list = [[] for i in range(self.NUM_TOPICS)]
        # for i in range(len(self.xcorpus)):
        #     for j in range(len(self.xcorpus[i])):
        #         if self.xcorpus[i][j] not in word_topic:
        #             word_topic[self.xcorpus[i][j]] = self.ycorpus[i][j]
        for i in range(len(self.xcorpus)):
            for j in range(len(self.xcorpus[i])):
                word = self.xcorpus[i][j]
                topic = self.ycorpus[i][j]
                if word not in word_topic_list[topic]:
                    word_topic_list[topic].append(word)
        # with open(out_path, "w") as f_out:
        #     for k, v in word_topic.items():
        #         print(k, v, file=f_out)

        for i in range(len(word_topic_list)):
            random.shuffle(word_topic_list[i])
        with open(out_path, "w") as f_out:
            for i, values in enumerate(word_topic_list):
                print(i, file=f_out)
                for v in values:
                    print(v, end=' ', file=f_out)
                print(file=f_out)


if __name__ == '__main__':
    if argv[1:] == ['test']:
        in_path = '../../../nlptutorial/test/07-train.txt'
        is_tqdm = False
    else:
        in_path = '../../../nlptutorial/data/wiki-en-documents.word'
        is_tqdm = True
    topic_model = Topic_Model(20, 2, 0.01, 0.01, is_tqdm)  # iteration_num, NUM_TOPICS, α, β
    topic_model.init(in_path)
    topic_model.sampling('out_sep_shuffle_test.txt')
