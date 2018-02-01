.. _sampling:

Sampling
--------

This example demonstrates a Latin Hypercube Sampling of a 4 parameter 5 response model using the :func:`lhs <matk.matk.matk.lhs>` function. 
The models of the parameter study are run using the :func:`run <matk.sampleset.SampleSet.run>` function. 
The generation of diagnostic plots is demonstrated using :func:`hist <matk.sampleset.hist>`, :func:`panels <matk.sampleset.SampleSet.panels>`, and :func:`corr <matk.sampleset.corr>`.

.. include:: sampling.rst

