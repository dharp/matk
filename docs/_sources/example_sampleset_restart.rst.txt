.. _sampleset_restart:

Sampleset Restart
-----------------

This example demonstrates restarting a sampleset run using a log file from a run where not all samples ran.
To imitate the case where not all samples ran in the first attempt, the log file is rewritten with 5 samples missing.
Then the *restart_logfile* option is used in the run command using the modified log file.
The result is that the second run attempt reads in the completed runs from the log file and then runs the incomplete runs.
The end result is the sampleset object contains the results of all samples

.. include:: sampleset_restart.rst

