.. _cluster:

Running on Cluster
------------------

This example demonstrates the same calibration as :ref:`calibrate`, but sets up the MATK model as an external simulator to demonstrate how to utilize cluster resources. Similar to the :ref:`ext-sim` example, the subprocess call (`<https://docs.python.org/2/library/subprocess.html>`_) method is used to make system calls to run the *model* and MATK's :func:`pest_io.tpl_write <matk.pest_io.tpl_write>` is used to create model input files with parameters in the correct locations. The pickle package (`<https://docs.python.org/2/library/pickle.html>`_) is used for I/O of the model results between the external simulator (sine.tpl) and the MATK model. This example is designed for a cluster using slurm and moab and will have to be modified for use on clusters using other resource managers. 

:download:`DOWNLOAD SCRIPT <../../examples/lmfit_cluster/calibrate_sine_lmfit_cluster.py>`

:download:`DOWNLOAD MODEL TEMPLATE FILE <../../examples/lmfit_cluster/sine.tpl>`

.. literalinclude:: ../../examples/lmfit_cluster/calibrate_sine_lmfit_cluster.py

Template file used by :func:`pest_io.tpl_write <matk.pest_io.tpl_write>`. Note the header **ptf %** and parameter locations indicated by **%** in the file. 

.. literalinclude:: ../../examples/lmfit_cluster/sine.tpl


