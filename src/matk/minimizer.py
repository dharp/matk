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
            self.__set_internal_parvalues(fvars)
            self.update_constraints()
            pardict = dict(zip(self._parent.parnames, self._parent.parvalues))
            self.userfcn(pardict=pardict, *self.userargs, **self.userkws)
        else:
            self.__set_internal_parvalues(self.vars)
            self.userfcn(*self.userargs, **self.userkws)
        self.nfev = self.nfev + 1
        out = self._parent.residuals
        if hasattr(self.iter_cb, '__call__'):
            self.iter_cb(self.params, self.nfev, out,
                         *self.userargs, **self.userkws)
        return out

    def __get_internal_parvalues(self, fvars):
        parvalues = []
        for varname, val in zip(self.var_map, fvars):
            par = self.params[varname]
            parvalues.append(par.from_internal(val))
        return numpy.array(parvalues)

    def __set_internal_parvalues(self, fvars):
        for varname, val in zip(self.var_map, fvars):
            par = self.params[varname]
            par.value = par.from_internal(val)

    def __jacobian( self, h=1.e-3, cpus=1, workdir_base=None,
                    save=True, reuse_dirs=False ):
        ''' Numerical Jacobian calculation

            :param h: Parameter increment, single value or array with npar values
            :type h: fl64 or ndarray(fl64)
            :returns: ndarray(fl64) -- Jacobian matrix
        '''
        # Collect parameter sets
        a = self.vars
        # If current simulated values are associated with current parameter values...
        if self._parent._current:
            sims = self._parent.simvalues
        if isinstance(h, (tuple,list)):
            h = numpy.array(h)
        elif not isinstance(h, numpy.ndarray):
            h = numpy.ones(len(a))*h
        hlmat = numpy.identity(len(self.vars))*-h
        humat = numpy.identity(len(self.vars))*h
        hmat = numpy.concatenate([hlmat,humat])
        parset = []
        for hs in hmat:
            int_pars = self.__get_internal_parvalues(hs+a)
            parset.append(int_pars)
        parset = numpy.array(parset)
        self._parent.create_sampleset(parset,name='_jac_')

        self._parent.sampleset['_jac_'].run( cpus=cpus, verbose=False,
                         workdir_base=workdir_base, save=save, reuse_dirs=reuse_dirs )
        # Perform simulations on parameter sets
        obs = self._parent.sampleset['_jac_'].responses.values
        a_ls = obs[0:len(a)]
        a_us = obs[len(a):]
        J = []
        for a_l,a_u,hs in zip(a_ls,a_us,h):
            J.append((a_l-a_u)/(2*hs))
        self._parent.parvalues = a
        # If current simulated values are associated with current parameter values...
        if self._parent._current:
            self._parent._set_simvalues(sims)
        return numpy.array(J).T

    def calibrate( self, cpus=1, maxiter=100, lambdax=0.001, minchange=1.0e-16, minlambdax=1.0e-6, verbose=False,
                  workdir=None, reuse_dirs=False, h=1.e-6):
        """ Calibrate MATK model using Levenberg-Marquardt algorithm based on 
            original code written by Ernesto P. Adorio PhD. 
            (UPDEPP at Clarkfield, Pampanga)

            :param cpus: Number of cpus to use
            :type maxiter: int
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
        #a = self.vars # Initial parameter values
        besta = self.vars # Best parameters start as current parameters
        self.__residual(self.vars)
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
                J = self.__jacobian( cpus=cpus, h=h )
                # Compute Hessian
                JtJ = numpy.dot(J.T,J)
                if (lambdax == 0.0) :
                    break
                # Form RHS beta vector
                r = numpy.array(self.__residual(self.vars))
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
                newa = self.vars + delta
                # and new sum of squares
                self.__residual(newa)
                newSS = self._parent.ssr
                if verbose: print "newSS = ", newSS
                # Update current parameter vector?
                if (newSS < bestSS):
                    if verbose: print "improved values found!"
                    besta  = newa
                    bestSS = newSS
                    bestJtJ = JtJ
                    self.vars = newa
                    iscomp = True
                    if verbose:
                        print "new a:"
                        for x in self.vars:
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
        self.__residual(besta)
        if verbose:
            print 'Parameter: '
            print self._parent.parvalues
            print 'SSR: '
            print self._parent.ssr
            print 'Flag: ', flag
            try:
                Cov = numpy.linalg.inv(JtJ)
            except numpy.linalg.linalg.LinAlgError as err:
                print "Warning: Unable to compute covariance - " + str(err)   
            else:
                print 'Cov: '
                print Cov


