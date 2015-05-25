Getting Started
===============

The easiest way to get started with MATK is to open an ipython/python terminal and copy and paste the example code below into your terminal. After that, the :ref:`examples` section can be explored for additional ideas of how MATK can facilitate your model analysis. 

.. toctree::

Load MATK module
----------------

Start by importing the MATK module:

.. testcode::

    import matk

We'll use numpy and some scipy functions also, so load those modules:

.. testcode::

    import numpy
    from scipy import arange, randn, exp

Define Model
------------

To perform a model analysis, MATK needs a model in the form of a python function that accepts a dictionary of parameter values keyed by parameter names as the first argument and returns an integer, float, array, or dictionary of model results keyed by result names. 
For demonstration purposes, we'll define a simple python function that computes a summation of exponential function and returns the results as an array:

.. testcode::

    def dbexpl(p):
        t=arange(0,100,20.)
        y = (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
        return y

Of course the python function can be much more complicated, including advanced scipy (`<http://www.scipy.org>`_) functions or calls to external programs (e.g. :ref:`ext-sim`). 

To test that the function does what you expect it to do, you can create a parameter dictionary and pass it to the function:

.. testcode::
    
    pars = {'par1':0.5,'par2':0.1,'par3':0.5,'par4':0.1}
    print dbexpl(pars)

.. testoutput::

    [  1.00000000e+00   1.35335283e-01   1.83156389e-02   2.47875218e-03
       3.35462628e-04]

This is only to test the function, MATK will be generating the parameter dictionaries during the model analysis.

If you haven't added observations to your MATK object, the first time the MATK *model* is called, observations are automatically created and given default names of *obs1*, *obs2*, ..., *obsN*, where *N* is the number of observations. If you have added observations to your MATK object and your *model* returns an array, the array ordering must match the order in which you added observations. If your *model* returns a dictionary, the order is irrelevant. However, if your *model* returns a dictionary and you have defined observations in your MATK object, your *model* must return a dictionary with keys that match all the defined observations. An example of *dbexpl* that returns a dictionary is:

.. testcode::

    def dbexpl(p):
        t=arange(0,100,20.)
        y = (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
        ydict = dict([('obs'+str(i+1), v)  for i,v in enumerate(y)])
        return ydict

.. testcode::
    
    print dbexpl(pars)

.. testoutput::

    {'obs4': 0.0024787521766663585, 'obs5': 0.00033546262790251185, 'obs2': 0.1353352832366127, 'obs3': 0.018315638888734179, 'obs1': 1.0}

Note that the dictionary is out of order. As mentioned above, this is irrelevant since the keys indicate to which observation the values are associated.

To maintain the order of the returned dictionary, you can return an *OrderedDict* from the *collections* package (`<https://docs.python.org/2/library/collections.html>`_) included in MATK:

.. testcode::

    from matk.ordereddict import OrderedDict

    def dbexpl(p):
        t=arange(0,100,20.)
        y = (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
        ydict = OrderedDict([('obs'+str(i+1), v)  for i,v in enumerate(y)])
        return ydict

.. testcode::
    
    print dbexpl(pars)

.. testoutput::

    OrderedDict([('obs1', 1.0), ('obs2', 0.1353352832366127), ('obs3', 0.018315638888734179), ('obs4', 0.0024787521766663585), ('obs5', 0.00033546262790251185)])

As mentioned, while dictionary ordering may be desirable, it is not required by MATK.

Create MATK Object
------------------

Create an instance of the MATK class specifying the function created above as the *model* using a keyword argument:

.. testcode::

    p = matk.matk(model=dbexpl)

Add Parameters
--------------

Add parameters to the model analysis matching those in the MATK model:

.. testcode::

    p.add_par('par1',min=0,max=1,value=0.5)
    p.add_par('par2',min=0,max=0.2,value=0.1)
    p.add_par('par3',min=0,max=1,value=0.5)
    p.add_par('par4',min=0,max=0.2,value=0.1)

Check current parameter values:

.. testcode::

   print p.parvalues

.. testoutput::

    [0.5, 0.1, 0.5, 0.1]    

and parameter names:

.. testcode::

    print p.parnames

.. testoutput::

    ['par1', 'par2', 'par3', 'par4']
    
and other useful information:

.. testcode::

    print p.parmins

.. testoutput::

    [0, 0, 0, 0] 

.. testcode::

    print p.parmaxs

.. testoutput::

    [1, 0.2, 1, 0.2]

You can also access parameters using the MATK *pars* dictionary:

.. testcode::
    
    print p.pars

.. testoutput::

    OrderedDict([('par1', <Parameter 'par1', 0.5, bounds=[0:1]>), ('par2', <Parameter 'par2', 0.1, bounds=[0:0.2]>), ('par3', <Parameter 'par3', 0.5, bounds=[0:1]>), ('par4', <Parameter 'par4', 0.1, bounds=[0:0.2]>)])

Individual parameters can be accessed using the *pars* dictionary as:

.. testcode::
    
    print p.pars['par1']

.. testoutput::

    <Parameter 'par1', 0.5, bounds=[0:1]>

.. testcode::
    
    print p.pars['par1'].value

.. testoutput::

    0.5

.. testcode::
    
    print p.pars['par1'].min

.. testoutput::

    0

.. testcode::
    
    print p.pars['par1'].max

.. testoutput::

    1

Add Observations
----------------

Observations are values that you want to compare model results to. These may be measurements that have been collected from the system you are modeling. Let's assume we have the following measurements for our system:

.. testcode::

    observations = [ 1., 0.14, 0.021, 2.4e-3, 3.4e-4]

We'll add these to the model analysis using generic names as:

.. testcode::

    for i,o in enumerate(observations): p.add_obs( 'obs'+str(i+1), value=o)

In cases where there are no observations (measurements) for comparison, the *value* keyword argument can be omitted. 
Check observation values and names:

.. testcode::

    print p.obsvalues

.. testoutput::

    [1.0, 0.14, 0.021, 0.0024, 0.00034]

.. testcode::

    print p.obsnames

.. testoutput::

    ['obs1', 'obs2', 'obs3', 'obs4', 'obs5']    

Similar to parameters, observations can be accessed using the *obs* dictionary:

.. testcode::
    
    print p.obs['obs1'].value

.. testoutput::

    1.0





