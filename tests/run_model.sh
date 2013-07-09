#!/bin/tcsh -f

dprepro $1 exp_model_dkt.tpl exp_model.in
python exp_model.py >& /dev/null

