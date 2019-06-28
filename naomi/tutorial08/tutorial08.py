from train_rnn import TrainRNN
from test_rnn import TestRNN

if __name__ == "__main__":
    epoch = 3
    node = 20

    train = '../../test/05-train-input.txt'
    test = '../../test/05-test-input.txt'

    train = '../../data/wiki-en-train.norm_pos'
    test = '../../data/wiki-en-test.norm'

    rnn = TrainRNN(epoch, node)
    net, word_ids, pos_ids = rnn.train(train)

    rnn = TestRNN(net, word_ids, pos_ids)
    rnn.test(test)