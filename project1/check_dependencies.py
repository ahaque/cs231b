import sys
import subprocess

try:
    print 'Checking for sklearn...'
    import sklearn
    print 'sklearn found.'
except ImportError:
    print 'You do not have sklearn installed. Please use pip or your favorite method to install sklearn for python.'

try:
    print 'Checking for numpy...'
    import numpy
    print 'numpy found.'
except ImportError:
    print 'You do not have numpy installed. Please use pip or your favorite method to install numpy for python.'

try:
    print 'Checking for matplotlib...'
    import matplotlib
    print 'matplotlib found.'
except ImportError:
    print 'You do not have MatPlotLib installed. Please use pip or your favorite method to install matplotlib for python.'

try:
    print 'Checking for argparse...'
    import argparse
    print 'argparse found.'
except ImportError:
    print 'You do not have argparse installed. Please use pip or your favorite method to install argparse for python.'

try:
    print 'Checking for Cython...'
    import Cython
    print 'Cython found.'
except:
    print 'You do not have Cython installed. Please use pip or your favorite method to install Cython for python.'

try:
    print 'pymaxflow...'
    import third_party.pymaxflow.pymaxflow as pymaxflow
    print 'pymaxflow found.'
except ImportError:
    print 'You have not built pymaxflow yet. Going to build it for you now...'
    process = subprocess.Popen(["sh", "./third_party/pymaxflow/build.sh"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()[0]
    print ''
    print 'pymaxflow built and linked'