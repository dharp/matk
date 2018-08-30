##!/usr/bin/python
# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
# Name:        lhs.py
# Project:  Bayesian-Inference
# Purpose:     
#
# Author:      Flávio Codeço Coelho<fccoelho@gmail.com>
#
# Created:     2008-11-26
# Copyright:   (c) 2008 by the Author
# Licence:     GPL
#-----------------------------------------------------------------------------
#__docformat__ = "restructuredtext en"
#from pylab import plot, figure,hist,show, savefig, legend
import scipy.stats as stats
import numpy
from numpy.linalg import cholesky,inv
from numpy.random import uniform, shuffle

def lhsFromSample(sample,siz=100):
    """
    Latin Hypercube Sample from a set of values.
    For univariate distributions only

    :Parameters:
        - `sample`: list, tuple of array
        - `siz`: Number or shape tuple for the output sample
    """
    #TODO: add support to correlation restricted multivariate samples
    if not isinstance(sample, (list,tuple,numpy.ndarray)):
        raise TypeError('sample is not a list, tuple or numpy vector')
    n = siz
    if isinstance(siz,(tuple,list)):
        n=numpy.product(siz)
    perc = numpy.arange(0,100.,100./n)
    shuffle(perc)
    smp = [stats.uniform(i,100./n).rvs() for i in perc]
    v = numpy.array([stats.scoreatpercentile(sample,p) for p in smp])
    if isinstance(siz,(tuple,list)):
        v.shape = siz
    return v

def lhsFromDensity(kde,siz=100):
    """
    LHS sampling from a variable's Kernel density estimate.

    :Parameters:
        - `kde`: scipy.stats.kde.gaussian_kde object
        - `siz`: Number or shape tuple for the output sample
    """
    if not isinstance(kde,scipy.stats.kde.gaussian_kde):
        raise TypeError("kde is not a density object")
    if isinstance(siz,(tuple,list)):
        n=numpy.product(siz)
    s = kde.resample(n)
    v = lhsFromSample(s,n)
    if isinstance(siz,(tuple,list)):
        v.shape = siz
    return v


def lhs(dist, parms, siz=100, noCorrRestr=False, corrmat=None, seed=None):
    '''
    Latin Hypercube sampling of any distribution.
    dist is is a scipy.stats random number generator 
    such as stats.norm, stats.beta, etc
    parms is a tuple with the parameters needed for 
    the specified distribution.

    :Parameters:
        - `dist`: random number generator from scipy.stats module or a list of them.
        - `parms`: tuple of parameters as required for dist, or a list of them.
        - `siz` :number or shape tuple for the output sample
        - `noCorrRestr`: if true, does not enforce correlation structure on the sample.
        - `corrmat`: Correlation matrix
    '''
    if seed:
        numpy.random.seed( seed )
    if not isinstance(dist,(list,tuple)):
        dists = [dist]
        parms = [parms]
    else:
        assert len(dist) == len(parms)
        dists = dist
    indices=rank_restr(nvars=len(dists), smp=siz, noCorrRestr=noCorrRestr, Corrmat=corrmat)
    smplist = []
    for j,d in enumerate(dists):
        if not isinstance(d, (stats.rv_discrete,stats.rv_continuous)):
            raise TypeError('dist is not a scipy.stats distribution object')
        n=siz
        if isinstance(siz,(tuple,list)):
            n=numpy.product(siz)
        #force type to float for sage compatibility
        pars = tuple([float(k) for k in parms[j]])
        #perc = numpy.arange(1.,n+1)/(n+1)
        step = 1./(n)
        perc = numpy.arange(0, 1, step) #class boundaries
        s_pos = [uniform(i, i+ step) for i in perc[:]]#[i+ step/2. for i in perc[:]]
        v = d(*pars).ppf(s_pos)
        #print len(v), step, perc
        index=list(map(int,indices[j]-1))
        v = v[index]
        if isinstance(siz,(tuple,list)):
            v.shape = siz
        smplist.append(v)
    if len(dists) == 1:
        return smplist[0]
    return numpy.array(smplist).transpose()

def rank_restr(nvars=4, smp=100, noCorrRestr=False, Corrmat=None):
    """
    Returns the indices for sampling variables with  
    the desired correlation structure.
    
    :Parameters:
        - `nvars`: number of variables
        - `smp`: number of samples
        - `noCorrRestr`: No correlation restriction if True
        - `Corrmat`: Correlation matrix. If None, assure uncorrelated samples.
    """
    if isinstance(smp,(tuple,list)):
            smp=numpy.product(smp)
    def shuf(s):
        """
        Shuffle a vector, making shure to make a copy of the original
        :param s: A vector of values
        :return: a list of arrays
        """
        s1=[]
        for i in range(nvars):
            shuffle(s)
            s1.append(s.copy())
        return s1
    if noCorrRestr or nvars ==1:
        inds = numpy.arange(smp)
        x = shuf(inds)
    else:
        if Corrmat == None:
            C=numpy.core.numeric.identity(nvars)
        else:
            if Corrmat.shape[0] != nvars:
                raise TypeError('Correlation matrix must be of rank %s'%nvars)
            C=numpy.matrix(Corrmat)
        s0=numpy.arange(1.,smp+1)/(smp+1.)
        s=stats.norm().ppf(s0)
        s1 = shuf(s)
        S=numpy.matrix(s1)
        P=cholesky(C)
        Q=cholesky(numpy.corrcoef(S))

        Final=S.transpose()*inv(Q).transpose()*P.transpose()
        x = [stats.stats.rankdata(Final.transpose()[i,]) for i in range(nvars)]
    return x

if __name__=='__main__':
    dist = stats.uniform,stats.uniform
    parms = (0,1.),(0,1.)
    print(lhs(dist,parms,siz=4))
    
    import pylab as P
    #dist = stats.norm
    dist = stats.beta
    #pars = (50,2)
    pars = (1,5) #beta
    b = lhs(dist,pars,1000)
    cm = numpy.array([[1,.8],[.8,1]])
    c=lhs([dist,dist], [pars,pars],2000,False, cm)
    #print stats.pearsonr(c[0],c[1]), stats.spearmanr(c[0],c[1])
    #P.hist(c[0],normed=1)#, label='c0 sample')
    P.scatter(c[0],c[1])
    #P.hist(c[1],normed=1)#, label='c1 sample')
    #print c[0].shape,c[1].shape
    n = dist(*pars).rvs(size=20)
    #hist(n.ravel(),facecolor='r',alpha =0.3,normed=1, label='Regular sample')
    #plot(numpy.arange(min(min(c),min(n)),max(max(c),max(n)),.1),dist(*pars).pdf(numpy.arange(min(min(c),min(n)),max(max(c),max(n)),.1)),label='PDF')
    #legend()
    #savefig('lhs.png',dpi=400)
#    lhs([stats.norm]*19,[(0,1)]*19,17,False,numpy.identity(19))
    P.show()
    
    

#TODO: Extend lhsFromSample to allow multivariate correlated sampling
