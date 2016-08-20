from matk import matk

# Create function
def fun(pars):
    o = (pars['x1'] - 1)**2 + (pars['x2'] - 2.5)**2
    return -o

# Set inequality constraints
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

p = matk(model=fun)

p.add_par('x1',min=0,value=2)
p.add_par('x2',min=0,value=0)
p.add_obs('obs1',value=0)
r = p.minimize(constraints=cons)


print "x1 should be 1.4: ", r['x'][0]
print "x2 should be 1.7: ", r['x'][1]

