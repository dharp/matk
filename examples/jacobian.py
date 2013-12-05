import matk
import numpy as np

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
a = np.array([0.7, 10., -0.4])
p.add_par('a0', value=0.7)
p.add_par('a1', value=10.)
p.add_par('a2', value=-0.4)

J = p.Jac()
