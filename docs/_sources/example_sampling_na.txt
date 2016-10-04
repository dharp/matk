Subsetting samples based on model output
----------------------------------------

This example demonstrates a case where a sample is subsetted based on model output.
This example builds on the :ref:`sampling` example.
Failed model calls are created by returning nans for some parameter combinations.
These samples are collected from the sampleset using the :func:`subset <matk.sampleset.subset>` function.
It is then easy to identify in the diagnostic plots where the problem is occurring.
This example demonstrates a Latin Hypercube Sampling of a 4 parameter 5 response model using the :func:`lhs <matk.matk.matk.lhs>` function. 
The models of the parameter study are run using the :func:`run <matk.sampleset.SampleSet.run>` function. 
The generation of diagnostic plots is demonstrated using :func:`hist <matk.sampleset.hist>`, :func:`panels <matk.sampleset.SampleSet.panels>`, and :func:`corr <matk.sampleset.corr>`.

.. include:: sampling_na.rst

