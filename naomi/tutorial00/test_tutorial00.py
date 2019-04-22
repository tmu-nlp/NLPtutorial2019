import unittest
import os
import filecmp
from tutorial00 import CountTokenFreq


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestTuto00(unittest.TestCase):
    def test_tuto00(self):
        print("test tutorial 00")
        in_file=os.path.join(THIS_DIR,os.pardir,os.pardir,'test/00-input.txt')
        freq_count_file = CountTokenFreq(in_file,'count_token.txt')
        ans_freq_count_file=os.path.join(THIS_DIR,os.pardir,os.pardir,'test/00-answer.txt')
        self.assertTrue(filecmp.cmp(ans_freq_count_file,freq_count_file,shallow=True))

if __name__=='__main__':
    unittest.main()
