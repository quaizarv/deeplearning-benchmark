Setup
=====

You will need python2.7, Scipy stack (including Numpy) and Theano
installed.

Follow the instructions at http://www.scipy.org/install.html to install
numpy and scipy

For MAC, a good option is to download and install canopy express from
Enthought at https://www.enthought.com/downloads/.

For insalling Theano, follow
http://deeplearning.net/software/theano/install.html. It also has
instructions to setup and enable Theano for GPU

For profiling in Theano, run with following command lines:
THEANO_FLAGS=mode=ProfileMode,device=gpu python <program_name>.py
THEANO_FLAGS=mode=ProfileMode,device=cpu python <program_name>.py

Add <dl-benchmark-dir>/common directory to your PYTHONPATH, where
<dl-benchmark-dir> is the name of the directory where you pulled
deeplearning-benchmark repositary:
  PYTHONPATH=<deeplearning-benchmark-dir>/common/:$PYTHONPATH

Also download mnist dataset from http://yann.lecun.com/exdb/mnist/ into
a directory called mnist under <dl-benchmark-dir> directory

Source Code Layout
==================

The souce code is laid out in the following directories:

common
------

The API definition and implementation are contained in this
directory. numpyWrappers.py contains the API implementation in
Numpy. theanoWrappers.py contains the API implementatio in Theano

To switch between Numpy and Theano API implementations, edit wrappers.py
by commenting out the appropriate import statement.

arrayIndexMapping.py maps array indices and dimensions from numpy to fortran
format and vice-versa

Utils.py contains higher level functions built on top of the API
operations and are shared by the various DL algorithms. They may be
converted into API operations in future if optimizing all the operations
of a higher level function together makes more sense.

This directory also contains other utilities used by the DL algorithms.

cnn
---

This directory contains CNN implementation in python.

cnnTrain.py - contains the main routine for training the paramters

cnnCost.py - contains routines AE cost and gradient computation 

minFuncSGD.py - contains routine for performing Stochastic Gradient Computation
using momentum

To run the CNN algorithm, you can just enter the following on a unix shell
"python test.py"


ae
--

This directory contains CNN implementation in python.

aeTrain.py - contains the main routine for training an auto-encoder

sparseAECost.py - contains routines for AE cost and gradient computation 

To run the AE algorithm, you can just enter the following on a unix shell
python test.py".


fista
-----

buildDict.py - contains code for learning a dictionary of features for 
sparse coding using INRIA's SPAMS library 

fista.py - contains code for learning a dictionary of features for 
sparse coding using FISTA. 

For description of FISTA, please read the following papers:

  Learning Fast Approximations of Sparse Coding - Karol Gregor and Yann
  LeCun

  A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
  Problems - Amir Beck† and Marc Teboulle


