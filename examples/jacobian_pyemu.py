import sys,os
sys.path.append('/Users/dharp/source-mac/pyemu')
import matk
import pyemu
from mat_handler import matrix,cov
import numpy as np
import matplotlib.pyplot as plt

def fv(a):
    a0 = a['a0']
    a1 = a['a1']
    a2 = a['a2']
    X = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])
    out = a0 / (1. + a1 * np.exp( X * a2))
    return out
    #obsnames = ['obs'+str(i) for i in range(1,len(out)+1)]
    #return dict(zip(obsnames,out))


p = matk.matk(model=fv)
p.add_par('a0', value=0.7)
p.add_par('a1', value=10.)
p.add_par('a2', value=-0.4)

J = p.Jac()

print np.dot(J.T,J)

m = matrix(x=J,row_names=p.obsnames,col_names=p.parnames)
parcov = cov(np.linalg.inv(np.dot(J.T,J)),names=p.parnames)
obscov = cov(np.linalg.inv(np.dot(J,J.T)),names=p.obsnames)

la = pyemu.errvar(jco=m,parcov=parcov,obscov=obscov)

s = la.qhalfx.s
plt.plot(s.x)
plt.show()

ident_df = la.get_identifiability_dataframe(3)
print ident_df

