import math
from collections import defaultdict

transition = {}
emission = defaultdict(lambda:0)
possible_tags = {}

lam = 0.95
V = 1000000

# モデル読み込み
with open('model.txt','r') as model_file:
    for line in model_file:
        line = line.rstrip()
        type_, context, word, prob = line.split(' ') # contextはタグ
        possible_tags[context] = 1 # 可能なタグとして保存
        if type_ == 'T':
            transition[f'{context} {word}'] = float(prob)
        else:
            emission[f'{context} {word}'] = float(prob)

#test_path = '../../test/05-test-input.txt'
test_path = '../../data/wiki-en-test.norm'

with open(test_path, 'r') as test_file, open('my_answer.pos', 'w') as ans_file:
    for line in test_file:
        line = line.rstrip()

        # 前向きステップ
        words = line.split(' ')
        len_words = len(words)
        best_score = {}
        best_edge = {}
        best_score['0 <s>'] = 0 # <s>から始まる
        best_edge['0 <s>'] = None

        for i in range(len_words):
            for prev in possible_tags.keys():
                for next_ in possible_tags.keys():
                    if f'{i} {prev}' in best_score and f'{prev} {next_}' in transition:
                        # HMM遷移確率（タグの数は少ないので平滑化不要）
                        P_T = transition[f'{prev} {next_}']
                        # HMM生成確率（未知語を扱うため平滑化必要）
                        P_E = lam * emission[f'{next_} {words[i]}'] + (1 - lam)/V

                        score = best_score[f'{i} {prev}'] - math.log2(P_T) - math.log2(P_E)
                        if f'{i + 1} {next_}' not in best_score or best_score[f'{i + 1} {next_}'] > score: # 順番逆にするとダメ
                            best_score[f'{i + 1} {next_}'] = score
                            best_edge[f'{i + 1} {next_}'] = f'{i} {prev}'

        # </s>に対して同じ操作をを行う
        for tag in possible_tags.keys():
            if  f'{len_words} {tag}' in best_score and f'{tag} </s>' in transition:
                # HMM遷移確率（タグの数は少ないので平滑化不要）
                P_T = transition[f'{tag} </s>']
                score = best_score[f'{len_words} {tag}'] - math.log2(P_T)
                if f'{len_words + 1} </s>' not in best_score or best_score[f'{len_words + 1} </s>'] > score:
                    best_score[f'{len_words + 1} </s>'] = score
                    best_edge[f'{len_words + 1} </s>'] = f'{len_words} {tag}'

        # print(best_edge)

        # 後ろ向きステップ
        tags = []
        next_edge = best_edge[f'{len_words + 1} </s>']
        while next_edge != '0 <s>':
            # このエッジの品詞を出力に追加
            position, tag = next_edge.split(' ')
            tags.append(tag)
            next_edge = best_edge[next_edge]
        print(' '.join(tags[::-1]), file=ans_file)
