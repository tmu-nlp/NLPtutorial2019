# test-perceptron.py
from train_perceptron import predict_all

if __name__ == "__main__":
    model_path = "tutorial05.model"
    test_path = "../../data/titles-en-test.word"
    output_path = "tutorial05.result"
    predict_all(model_path, test_path, output_path)
    # Accuracy = 90.967056%