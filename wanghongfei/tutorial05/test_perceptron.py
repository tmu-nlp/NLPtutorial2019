from train_perceptron import create_features
from train_perceptron import predict_one

w = {} 
with open('model','r') as model:
    for line in model:
        columm= line.strip().split('\t')
        w[columm[0]] =  int(columm[1])

test_path = './test/test'

with open(test_path,'r') as test_file:
    for line in test_file:
        line = line.rstrip()
        phi = create_features(str(line)) 
        predicted_label = predict_one(w, phi)
        print(predicted_label)

