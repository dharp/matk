External Simulator
------------------

This example demonstrates a simple parameter study with an external simulator (FEHM groundwater simulator) using the subprocess call (`<https://docs.python.org/2/library/subprocess.html>`_) function to make system calls. MATK's :func:`pest_io.tpl_write <matk.pest_io.tpl_write>` is used to create model input files with parameters in the correct locations. 

:download:`DOWNLOAD SCRIPT <../../examples/fehm/ext_sim.py>`

.. literalinclude:: ../../examples/fehm/ext_sim.py

Template file used by :func:`pest_io.tpl_write <matk.pest_io.tpl_write>`. Note the header **ptf %** and parameter location **%por0%** in the file. This example illustrates how to use an external simulator and other files required to run this example are not included.

.. literalinclude:: ../../examples/fehm/intact.tpl


