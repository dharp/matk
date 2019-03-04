.. _rbd_fast:

RBD Fast Sensitivity Analysis
-----------------------------

This example demonstrates a RBD-FAST sensitivity analysis using the :func:`rbd_fast <matk.matk.matk.rbd_fast>` function from SALib (`<https://github.com/SALib/SALib>`_). In this case, the sensitivity of the sum-of-squared errors (sse) to model parameters is evaluated. It is also possible to provide the name of an individual observation instead of the sse as an argument in the :func:`rbd_fast <matk.matk.matk.rbd_fast>` method (i.e., argument “obsname”).

.. include:: rbd_fast.rst

The results indicate that the model is most sensitive to “amp” followed by “decay”. The model is relatively insensitive to “shift” and “omega”.
