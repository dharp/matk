import unittest
from pesting import *
from pymads import *
import exp_model
from numpy import array

class TestSequenceFunctions(unittest.TestCase):

    #def setUp(self):

    def test_run_model(self):
        self.prob = read_pest('exp_model.pst')
        self.prob.run_model()
        SSE =  sum( array(self.prob.get_residuals())**2 )
        self.assertEqual(SSE,0.0, 'Value of SSE incorrect!')

    def test_calibrate(self):
        self.prob = read_pest('exp_model_calib.pst')
        x,cov_x,infodic,mesg,ier = self.prob.calibrate()
        SSE = sum( array(infodic['fvec'])**2)
        self.assertEqual(SSE, 0.0, 'Calibration unsuccessful!')
        

if __name__ == '__main__':
    unittest.main()