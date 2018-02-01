.. _sobol:

Sobol Sensitivity Analysis
--------------------------

This example demonstrates a Sobol sensitivity analysis using the :func:`saltelli <matk.matk.matk.saltelli>` sampler and :func:`sobol <matk.sampleset.SampleSet.sobol>` function from SALib (`<https://github.com/SALib/SALib>`_). 
In this case, the sensitivity of the sum-of-squared errors (sse) to model parameters is evaluated. It is also possible to provide the name of an individual observation instead of the sse as an argument in the :func:`sobol <matk.sampleset.SampleSet.sobol>` method (i.e., argument "obsname").

.. include:: sobol.rst


