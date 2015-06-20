.. _pyemu:

Linear Analysis of Calibration Using PYEMU
------------------------------------------

This example demonstrates a linear analysis of the :ref:`calibrate` example using the pyemu module (`<https://github.com/jtwhite79/pyemu>`_). Singular values from pyemu's eigenanalysis of the jacobian are plotted and identifiability of parameters are printed. The resulting identifiability values indicate that one of the parameters (**amp**) is significantly less identifiable than the others.

.. include:: calibrate_sine_lmfit_pyemu.rst


