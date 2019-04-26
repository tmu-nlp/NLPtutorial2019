import unittest
import os
import filecmp
from tutorial01 import trainunigram, testunigram



THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestTuto01(unittest.TestCase):
    
    def test_tuto01_train(self):
        testtrain = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-input.txt')
        anstrain = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-answer.txt')
        mymodel = '01-my-train-answer.txt'

        trainunigram(testtrain,mymodel)
        
        self.assertTrue(filecmp.cmp(anstrain,mymodel,shallow=True))

    def test_tuto01_test(self):
        testinput = os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-test-input.txt')
        anstest = os.path.join('01-my-test-answer.txt')
        myeval = '01-my-eval-answer.txt'
        model =  os.path.join(THIS_DIR,os.pardir,os.pardir,'test/01-train-answer.txt')
        testunigram(model, testinput, myeval)
        self.assertTrue(filecmp.cmp(anstest,myeval,shallow=True))
    
if __name__=='__main__':
    unittest.main()