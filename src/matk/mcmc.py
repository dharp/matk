from pymc import MCMC, MAP
from numpy import array

def mcmc(prob, nsample=100, modulename = 'model' ):
    try:
        mystr = "from " + modulename + " import model"
        exec(mystr)
    except:
        print 'cannot import', modulename
    M = MCMC( model(prob) )
    M.sample(nsample)
    return M

def map(prob, modulename = 'model' ):
    try:
        mystr = "from " + modulename + " import model"
        exec(mystr)
    except:
        print 'cannot import', modulename
    M = MAP( model(prob) )
    M.fit()
    return M

def create_pymc_model( prob, filename = 'model.py' ):
    
    try:
        f = open( filename, 'w' )
    except IOError:
        print 'cannot open', filename

    f.write( "from pymc import Uniform, deterministic, Normal\n\n" )
    f.write( "from numpy import array\n" )
    f.write( "def model(prob):\n" )
    f.write( "    #observations\n")
    f.write( "    obs = prob.get_observation_values()\n" )
    f.write( "\n    #priors\n")
    f.write( "    variables = []\n")
    f.write( "    sig = Uniform('sig', 0.0, 100.0, value=1.)\n")
    f.write( "    variables.append( sig )\n")
    for par in prob.get_parameters():
        if par.dist == 'uniform':
            f.write( "    " + str(par.name) + " = Uniform( '" + str(par.name) + "', " +  str(par.min) + ", " +  str(par.max) + ")\n")
            f.write( "    variables.append( " + str(par.name) + " )\n")
    f.write( "\n    #model \n")
    f.write( "    @deterministic() \n")
    f.write( "    def response( pars = variables, prob=prob ): \n")
    f.write( "        values = [] \n")
    f.write( "        for par in pars: \n")
    f.write( "            values.append( par ) \n")
    f.write( "        values = array( values ) \n")
    f.write( "        prob.set_parameters( values ) \n")
    f.write( "        prob.forward() \n")
    f.write( "        return prob.get_simvalues() \n")
    f.write( "\n    #likelihood \n")
    f.write( "    y = Normal('y', mu=response, tau=1.0/sig**2, value=obs, observed=True) \n")
    f.write( "    variables.append(y) \n")
    f.write( "\n    return variables \n")

    f.close()

