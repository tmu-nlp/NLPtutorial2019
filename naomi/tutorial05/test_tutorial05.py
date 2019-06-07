import unittest
import filecmp
from logging import getLogger, StreamHandler, DEBUG
from train_perceptron import train_perceptron, create_features


class TestTuto00(unittest.TestCase):

    def setUp(self):

        self.logger = getLogger(__name__)
        self.handler = StreamHandler()
        self.handler.setLevel(DEBUG)
        self.logger.setLevel(DEBUG)
        self.logger.addHandler(self.handler)
        self.logger.propagate = False

    def test_train_perceptron(self):

        # Test input for perceptron train
        train_input = '../../test/03-train-input.txt'

        # Test output model of perceptron train
        train_output = 'mymodel.txt'

        # Answer file
        train_answer = '../../test/03-train-answer.txt'

        # Train with perceptron
        train_perceptron(train_input, train_output)

        self.assertTrue(filecmp.cmp(train_output, train_answer, shallow=False))

    def test_create_features(self):

        ansdict = {'UNI:A': 1,
                   'UNI:site': 1,
                   'UNI:,': 2,
                   'UNI:Site': 1,
                   'UNI:in': 1,
                   'UNI:Maizuru': 1,
                   'UNI:Kyoto': 1}  
        # Create features
        outdict = create_features('A site , Site in Maizuru , Kyoto')

        self.assertTrue(ansdict == outdict)


if __name__ == '__main__':
    unittest.main()
