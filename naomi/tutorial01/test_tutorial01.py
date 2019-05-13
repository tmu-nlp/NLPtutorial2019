import unittest
import os
import filecmp
from tutorial01 import trainunigram, testunigram


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTuto01(unittest.TestCase):

    def test_tuto01_train(self):
        inp = 'test/01-train-input.txt'
        ans = 'test/01-train-answer.txt'
        testtrain = os.path.join(THIS_DIR, os.pardir, os.pardir, inp)
        anstrain = os.path.join(THIS_DIR, os.pardir, os.pardir, ans)
        mymodel = '01-my-train-answer.txt'

        trainunigram(testtrain, mymodel)

        self.assertTrue(filecmp.cmp(anstrain, mymodel, shallow=True))

    def test_tuto01_test(self):
        inp = 'test/01-test-input.txt'
        testinput = os.path.join(THIS_DIR, os.pardir, os.pardir, inp)
        anstest = os.path.join('01-my-test-answer.txt')
        myeval = '01-my-eval-answer.txt'
        train_ans = 'test/01-train-answer.txt'
        model = os.path.join(THIS_DIR, os.pardir, os.pardir, train_ans)
        testunigram(model, testinput, myeval)
        self.assertTrue(filecmp.cmp(anstest, myeval, shallow=True))


if __name__ == '__main__':
    unittest.main()
