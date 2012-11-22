import unittest
from pesting import *
from pymads import *
import exp_model
from numpy import array

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.prob = read_pest('exp_model.pst')

    def test_run_model(self):
        self.prob.run_model()
        SSE =  sum( array(self.prob.get_residuals())**2 )
        self.assertEqual(SSE,0.0, 'SSE should be zero!')

    def test_calibrate(self):
        self.calib_prob = read_pest('exp_model_calib.pst')
        x,cov_x,infodic,mesg,ier = self.calib_prob.calibrate()
        SSE = sum( array(infodic['fvec'])**2)
        self.assertEqual(SSE, 0.0, 'Calibration should result in SSE = 0.0!')

    def test_sample(self):
        s = self.prob.get_samples(1)
        lb = self.prob.get_lower_bounds()
        ub = self.prob.get_upper_bounds()
        self.assertTrue( (s >= lb).any() and (s <= ub).any(), 'Sample outside parameter bounds!' )

if __name__ == '__main__':
    unittest.main()