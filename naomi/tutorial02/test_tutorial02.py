import unittest
import os
import filecmp
from tutorial02 import trainbigram

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTuto01(unittest.TestCase):

    def test_tuto01_train(self):
        inp = 'test/02-train-input.txt'
        ans = 'test/02-train-answer.txt'
        testtrain = os.path.join(THIS_DIR, os.pardir, os.pardir, inp)
        anstrain = os.path.join(THIS_DIR, os.pardir, os.pardir, ans)
        mymodel = '02-my-train-answer.txt'

        trainbigram(testtrain, mymodel)

        self.assertTrue(filecmp.cmp(anstrain, mymodel, shallow=True))


if __name__ == '__main__':
    unittest.main()
