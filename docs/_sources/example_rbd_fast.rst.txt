.. _rbd_fast:

RBD-FAST Sensitivity Analysis
-----------------------------

This example demonstrates a RBD-FAST sensitivity analysis using the :func:`rbd_fast <matk.sampleset.SampleSet.rbd_fast>` function from SALib (`<https://github.com/SALib/SALib>`_). 
In this case, the sensitivity of the sum-of-squared errors (sse) to model parameters is evaluated. It is also possible to provide the name of an individual observation instead of the sse as an argument in the :func:`rbd_fast <matk.sampleset.SampleSet.rbd_fast>` method (i.e., argument "obsname").

.. include:: rbd_fast.rst


