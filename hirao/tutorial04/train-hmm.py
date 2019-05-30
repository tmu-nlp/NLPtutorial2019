# train-hmm.py
from collections import defaultdict

# train_input_path = "../../test/05-train-input.txt"
train_input_path = "../../data/wiki-en-train.norm_pos"
model_path = "tutorial04.txt"

# モデル読み込み
transition = defaultdict(lambda: 0)
emission = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)

with open(train_input_path) as f, open(model_path, mode="w") as fw:
    for line in f:
        word_tags = line.split()
        previous = "<s>"
        possible_tags[previous] += 1
        for word, tag in [x.split("_") for x in word_tags]:
            transition[f"{previous} {tag}"] += 1
            possible_tags[tag] += 1
            emission[f"{tag} {word}"] += 1
            previous = tag
        transition[f"{previous} </s>"] += 1
    for key, value in transition.items():
        previous, word = key.split()
        output = f"T {key} {value/possible_tags[previous]}"
        fw.write(output + "\n")
    for key, value in emission.items():
        previous, word = key.split()
        output = f"E {key} {value/possible_tags[previous]}"
        fw.write(output + "\n")