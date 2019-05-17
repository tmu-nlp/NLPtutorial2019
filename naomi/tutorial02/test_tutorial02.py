import unittest
import os
import filecmp
from tutorial02 import trainbigram

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTuto02(unittest.TestCase):

    def test_trainbigram(self):
        # 学習するテストデータ
        train = '../../test/02-train-input.txt'

        # trainbigramのアウトプットモデル
        mymodel = 'test-model.txt'

        # 比較する正しいモデル
        ansmodel = '../../test/02-train-answer.txt'

        trainbigram(train, mymodel)

        self.assertTrue(filecmp.cmp(ansmodel, mymodel, shallow=True))


if __name__ == '__main__':
    unittest.main()
