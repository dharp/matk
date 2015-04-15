# PyTOUGH setup script
from distutils.core import setup
import os

os.chdir('src')

setup(name='MATK',
	version='0.0.0',
	description='Model Analysis ToolKit (MATK) - Python toolkit for model analysis',
	author='Dylan R. Harp',
	author_email='dharp@lanl.gov',
	url='matk.lanl.gov',
	license='LGPL',
	packages=[
		'matk',
		'matk.emcee',
		'matk.lmfit',
		'matk.lmfit.uncertainties',
		'matk.pyDOE'],
	py_modules=['matk','lhs','parameter','sampleset','minimizer','pest_io','ordereddict','observation','__init__'],
	)
