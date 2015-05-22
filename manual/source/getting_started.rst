Getting Started
===============

.. toctree::

Load MATK module
----------------

Start by importing the matk module:

.. testcode::

    import matk

Define Model
------------

Define a model as a python function:

.. testcode::

    def dbexpl(p):
        t=arange(0,100,20.)
        y =  (p['par1']*exp(-p['par2']*t) + p['par3']*exp(-p['par4']*t))
        return y

Create MATK Object
------------------

Create an instance of the MATK class:

.. testcode::

    p = matk.matk(model=dbexpl)

Add Parameters
--------------

Add some parameter to the model analysis:

.. testcode::

    p.add_par('par1',min=0,max=1)
    p.add_par('par2',min=0,max=0.2)
    p.add_par('par3',min=0,max=1)
    p.add_par('par4',min=0,max=0.2)

Do a test:

.. testcode::

    print 1

.. testoutput::

    1

