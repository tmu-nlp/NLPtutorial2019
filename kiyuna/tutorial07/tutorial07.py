from train_nn import *
from test_nn import *
import subprocess

train_path = '../../data/titles-en-train.labeled'
train_nn(train_path, layer_num=1, node_num=2, epoch_num=1, λ=0.1)

test_path = '../../data/titles-en-test.word'
out_path = './out.txt'
test_nn(test_path, out_path)

script_path = '../../script/grade-prediction.py'
ans_path = '../../data/titles-en-test.labeled'
subprocess.run(f'{script_path} {ans_path} {out_path}'.split())


''' RESULT
Accuracy = 92.915338%
'''
