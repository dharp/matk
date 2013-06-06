from pymc import Uniform, deterministic, Normal

from numpy import array
def model(prob):
    #observations
    obs = prob.get_observation_values()

    #priors
    variables = []
    sig = Uniform('sig', 0.0, 100.0, value=1.)
    variables.append( sig )
    a1 = Uniform( 'a1', 0.0, 5.0)
    variables.append( a1 )
    k1 = Uniform( 'k1', 0.01, 2.0)
    variables.append( k1 )
    a2 = Uniform( 'a2', 0.0, 5.0)
    variables.append( a2 )
    k2 = Uniform( 'k2', 0.01, 2.0)
    variables.append( k2 )

    #model 
    @deterministic() 
    def response( pars = variables, prob=prob ): 
        values = [] 
        for par in pars: 
            values.append( par ) 
        values = array( values ) 
        prob.set_parameters( values ) 
        prob.forward() 
        return prob.get_sim_values() 

    #likelihood 
    y = Normal('y', mu=response, tau=1.0/sig**2, value=obs, observed=True) 
    variables.append(y) 

    return variables 
