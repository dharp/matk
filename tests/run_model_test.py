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
        obs = []
        sims = []
        for obsgrp in self.prob.obsgrp:
            for observation in obsgrp.observation:
                obs.append( observation.value )
                sims.append( observation.sim_value )
        obs = array(obs)
        sims = array(sims)
        res =  sum( (obs - sims)**2 )
        self.assertEqual(res,0.0, 'Value of SSE incorrect!')

if __name__ == '__main__':
    unittest.main()