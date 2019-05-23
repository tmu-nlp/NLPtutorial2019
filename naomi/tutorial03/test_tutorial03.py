import unittest
import filecmp
from tutorial03 import load, viterbi
from logging import getLogger, StreamHandler, DEBUG


# https://qiita.com/amedama/items/b856b2f30c2f38665701
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


class TestTuto03(unittest.TestCase):

    def test_trainbigram(self):
        # 読み込むモデル
        model = '../../test/04-model.txt'

        # 読み込むデータ
        inputf = '../../test/04-input.txt'

        # 正解の分割されたデータ
        answakachi = '../../test/04-answer.txt'

        # 分割されたデータ（アウトプット）
        mywakachi = 'myanswer.txt'

        # unigram probabilityの読み込み
        P = load(model)

        with open(inputf, 'r', encoding='utf-8') as fin:
            with open(mywakachi, 'w+', encoding='utf-8') as out:
                for line in fin:
                    words = viterbi(line.rstrip(), P)
                    text = ' '.join(words)
                    print(text, file=out)
                    logger.debug(text)

        self.assertTrue(filecmp.cmp(answakachi, mywakachi, shallow=False))


if __name__ == '__main__':
    unittest.main()
