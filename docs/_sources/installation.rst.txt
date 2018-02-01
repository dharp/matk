Obtaining MATK
==============

.. toctree::

Downloading MATK
----------------

MATK can be obtained by:

1. Using git::

    git clone https://github.com/dharp/matk

2. Clicking `here <https://github.com/dharp/matk/archive/master.zip>`_

3. Going to `<https://github.com/dharp/matk>`_ and clicking on the **Download ZIP** button on the right 

Installing MATK
---------------

To install MATK, enter the main directory in a terminal and ::

    python setup.py install

Depending on your system setup and privileges, you may want to do this as root (\*nix and mac systems):: 

    sudo python setup.py install

or as user::

    python setup.py install --user

If all these fail, you can set your PYTHONPATH to point to the MATK *src* directory

1. bash::

    export PYTHONPATH=/path/to/matk/src

2. tcsh::

    setenv PYTHONPATH /path/to/matk/src

3. Windows, I have no clue!

Testing installation
--------------------

To test that the MATK module is accessible, open a python/ipython terminal and

.. code-block:: python

    import matk

If the MATK module is accessible, this will load without an error.

For more in depth analysis of MATK functionality on your system, the test suite can be run by entering the MATK *tests* directory in a terminal and::

    python -W ignore matk_unittests.py -v

Test results will be printed to the terminal.

