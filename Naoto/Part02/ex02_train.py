def count_bigram_words(document: str, model_: str, counts: {}, context_counts: {}):
    with open(model_, "w") as model:
        for line in document:
            words = ["<s>"] + line.rstrip().split() + ["</s>"]
            for i in range(1, len(words)):
                counts[words[i-1] + " " + words[i]] += 1
                context_counts[words[i-1]] += 1
                counts[words[i]] += 1
                context_counts[""] += 1
        # print(counts)
        # print(context_counts)
        for ngram, count in sorted(counts.items()):
            # print(ngram)
            # print(count)
            context = ngram.split()
            # print(context)
            # print()
            if len(context) == 2:
                probability = counts[ngram]/context_counts[context[0]]
            else:
                probability = counts[ngram]/context_counts[""]
            model.write(ngram + " " + str(probability) + "\n")


if __name__ == '__main__':
    from collections import defaultdict
    train_ = "wiki-en-train.word"
    model_ = "model.txt"
    counts_txt = "counts.txt"
    # train_ = "02-train-input.txt"
    # model_ = "02-train-output.txt"
    train = open(train_, "r")
    
    counts = defaultdict(lambda: 0)
    context_counts = defaultdict(lambda: 0)
    count_bigram_words(train, model_, counts, context_counts)
    train.close()