.. _external:

External Simulator (Python script)
----------------------------------

This example demonstrates the same calibration as :ref:`calibrate`, but sets up the MATK model as an python script to demonstrate how to use an external simulator. 
Similar to the :ref:`ext-sim` example, the subprocess call (`<https://docs.python.org/2/library/subprocess.html>`_) method is used to make system calls to run the *model* and MATK's :func:`pest_io.tpl_write <matk.pest_io.tpl_write>` is used to create model input files with parameters in the correct locations. 
The pickle package (`<https://docs.python.org/2/library/pickle.html>`_) is used for I/O of the model results between the external simulator (sine.tpl) and the MATK model. 

:download:`DOWNLOAD SCRIPT <../../examples/lmfit_external/calibrate_sine_lmfit_external.py>`

:download:`DOWNLOAD MODEL TEMPLATE FILE <../../examples/lmfit_external/sine.tpl>`

.. plot:: ../../examples/lmfit_external/calibrate_sine_lmfit_external.py
    :include-source:

Template file used by :func:`pest_io.tpl_write <matk.pest_io.tpl_write>`. Note the header **ptf %** and parameter locations indicated by **%** in the file. 

.. literalinclude:: ../../examples/lmfit_external/sine.tpl


