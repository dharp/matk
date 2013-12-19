from lmfit.minimizer import Minimizer as LmfitMinimizer
import numpy
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

class Minimizer(LmfitMinimizer):
    err_nonparam = \
     "params must be an OrderedDict"

    def __init__(self, parent, fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **kws):
        super(Minimizer, self).__init__(parent.forward, parent.pars, fcn_args=fcn_args,
                 fcn_kws=fcn_args, iter_cb=iter_cb, scale_covar=scale_covar, **kws)
        self._parent = parent

    def __set_params(self, params):
        """ set internal self.params from a Parameters object or
        a list/tuple of Parameters"""
        if params is None or isinstance(params, OrderedDict):
            self.params = params
        #elif isinstance(params, (list, tuple)):
        #    _params = Parameters()
        #    for _par in params:
        #        if not isinstance(_par, Parameter):
        #            raise MinimizerException(self.err_nonparam)
        #        else:
        #            _params[_par.name] = _par
        #    self.params = _params
        else:
            raise MinimizerException(self.err_nonparam)

    def __residual(self, fvars=None):
        """
        residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables),
        this evaluates all parameters, including setting bounds and
        evaluating constraints, and then passes those to the
        user-supplied function to calculate the residual.
        """
        if not fvars is None:
            # set parameter values
            for varname, val in zip(self.var_map, fvars):
                # self.params[varname].value = val
                par = self.params[varname]
                par.value = par.from_internal(val)
            self.nfev = self.nfev + 1

            self.update_constraints()
            pardict = dict(zip(self._parent.parnames, self._parent.par_values))
            self.userfcn(pardict=pardict, *self.userargs, **self.userkws)
        else:
            self.userfcn(*self.userargs, **self.userkws)
        out = self._parent.residuals
        if hasattr(self.iter_cb, '__call__'):
            self.iter_cb(self.params, self.nfev, out,
                         *self.userargs, **self.userkws)
        return out



    def calibrate( self, maxiter=100, lambdax=0.001, minchange=1.0e-1, minlambdax=1.0e-6, verbose=False,
                  workdir=None, reuse_dirs=False):
        """ Calibrate MATK model using Levenberg-Marquardt algorithm based on 
            original code written by Ernesto P. Adorio PhD. 
            (UPDEPP at Clarkfield, Pampanga)

            :param maxiter: Maximum number of iterations
            :type maxiter: int
            :param lambdax: Initial Marquardt lambda
            :type lambdax: fl64
            :param minchange: Minimum change between successive ChiSquares
            :type minchange: fl64
            :param minlambdax: Minimum lambda value
            :type minlambdax: fl4
            :param verbose: If True, additional information written to screen during calibration
            :type verbose: bool
            :returns: best fit parameters found by routine
            :returns: best Sum of squares.
            :returns: covariance matrix
        """
        self.prepare_fit()
        
        n = len(self._parent.obs) # Number of observations
        m = len(self._parent.pars) # Number of parameters
        a = numpy.copy(self._parent.par_values) # Initial parameter values
        besta = a # Best parameters start as current parameters
        #self._parent.forward(workdir=workdir, reuse_dirs=reuse_dirs)
        self.__residual()
        bestSS = SS = self._parent.ssr # Sum of squared error
        Cov = None
        iscomp = True
        ncount = 0
        flag   = 0
        for p in range(1, maxiter+1):
            if verbose: print "marquardt(): iteration=", p
            # If iscomp, recalculate JtJ and beta
            if (iscomp) :
                # Compute Jacobian
                J = self._parent.Jac()
                # Compute Hessian
                JtJ = numpy.dot(J.T,J)
                if (lambdax == 0.0) :
                    break
                # Form RHS beta vector
                #pardict = dict(zip(self._parent.parnames, a))
                #self._parent.forward(pardict=pardict, workdir=workdir, reuse_dirs=reuse_dirs)
                r = numpy.array(self.__residual(a))
                beta = -numpy.dot(J.T,r)

            # Update A with new lambdax
            A = JtJ * (numpy.ones(m) + numpy.identity(m)*lambdax)

            # Solve for delta
            try:
                delta = numpy.linalg.solve(A, beta)
            except numpy.linalg.linalg.LinAlgError as err:
                print "Error: Unable to solve for update vector - " + str(err)
                break
            else:
                code=0
            totabsdelta = numpy.sum(numpy.abs(delta))
            if verbose:
                print "JtJ:"
                print JtJ
                try:
                    Cov = numpy.linalg.inv(JtJ)
                except numpy.linalg.linalg.LinAlgError as err:
                    print "Warning: Unable to compute covariance - " + err   
                else:
                    print 'Cov: '
                    print Cov
                print "beta = ", beta
                print "delta=", delta
                print "SS =",SS
                print "lambdax=", lambdax
                print "total abs delta=", totabsdelta
            if (code == 0):
                # Compute new parameters
                newa = a + delta
                # and new sum of squares
                #pardict = dict(zip(self._parent.parnames, newa))
                #self._parent.forward(pardict=pardict, workdir=workdir, reuse_dirs=reuse_dirs)
                self.__residual(newa)
                newSS = self._parent.ssr
                if verbose: print "newSS = ", newSS
                # Update current parameter vector?
                if (newSS < bestSS):
                    if verbose: print "improved values found!"
                    besta  = newa
                    bestSS = newSS
                    bestJtJ = JtJ
                    a = newa
                    #a = newa
                    iscomp = True
                    if verbose:
                        print "new a:"
                        for x in a:
                            print x
                        print
                    # Termination criteria
                    if (SS - newSS < minchange):
                        ncount+= 1
                        if (ncount == 2) :
                            lambdax  = 0.0
                            flag = 0
                            break
                    else :
                        ncount = 0
                        lambdax = 0.4 * lambdax  # after Nash
                        if (lambdax < minlambdax) :
                            flag = 3
                            break
                    SS = newSS
                else :
                    iscomp = False
                    lambdax = 10.0 * lambdax
                    ncount = 0
            else :
                flag = 1
                break
        if (flag == 0):
            if Cov is None: flag = 4
            if (p >= maxiter) :
                flag = 2
        #self._parent.par_values = besta
        #self._parent.forward( workdir=workdir, reuse_dirs=reuse_dirs)
        self.__residual(besta)
        if verbose:
            print 'Parameter: '
            print self._parent.par_values
            print 'SSR: '
            print self._parent.ssr
            try:
                Cov = numpy.linalg.inv(JtJ)
            except numpy.linalg.linalg.LinAlgError as err:
                print "Warning: Unable to compute covariance - " + err   
            else:
                print 'Cov: '
                print Cov


