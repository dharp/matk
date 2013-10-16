import pymc
import mcmc 
from pymads import *

p = pesting.read_pest('exp_model.pst')

mcmc.create_pymc_model(p)

M = mcmc.mcmc(p, nsample=10)

