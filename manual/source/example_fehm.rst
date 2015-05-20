External Simulator
------------------

This example demonstrates using MATK with an external simulator (FEHM groundwater simulator) using the :func:`~subprocess.call` function to make system calls. MATK's :func:`pest_io.tpl_write <matk.matk.pest_io.tpl_write>` is used to create model input files with parameters in the correct locations. 

:download:`DOWNLOAD SCRIPT <../../examples/fehm/ext_sim.py>`

.. literalinclude:: ../../examples/fehm/ext_sim.py

