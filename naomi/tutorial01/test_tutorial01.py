import unittest
import os
import filecmp
from tutorial01 import trainunigram, testunigram



THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestTuto01(unittest.TestCase):
    
    def test_tuto01_train(self):
        testtrain = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-input.txt')
        anstrain = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-answer.txt')
        mytrain = '01-my-train-answer.txt'

        trainunigram(testtrain,mytrain)
        
        self.assertTrue(filecmp.cmp(anstrain,mytrain,shallow=True))

    def test_tuto01_test(self):
        testinput = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-test-input.txt')
        anstest = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-test-answer.txt')
        mytest = '01-my-test-answer.txt'
        
        testunigram(testinput, mytest)

        self.assertTrue(filecmp.cmp(anstest,mytest,shallow=True))
    
if __name__=='__main__':
    unittest.main()