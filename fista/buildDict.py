""" This file contains code to learn a dictionary of features for Sparse Coding
    using the SPAMS package from INRIA
"""
import time
import spams
from scipy.sparse import csc_matrix
import numpy as np
from samplePatches import *
from displayNetwork import *

DICT_FILE = 'sparse_code_dict'

def sparse_codes(X, D, lamda):
  """ Map input data in X to sparse codes using dictionary D using
     Coordinate Descent from SPAM package

     Parameters:
      X:     Input data as (num_patches, input_dim) matrix
      D:     Distonary of features
  """
  # Map X and D to fortran array format
  X0 = np.asfortranarray(np.transpose(X))
  D0 = np.asfortranarray(np.transpose(D))

  # Get a sparse matrix
  A0 = csc_matrix(np.zeros((D.shape[0], X.shape[0])))
  out = spams.cd(X0, D0, A0,
                 lambda1 = lamda, mode = 2, itermax = 1000, tol = 0.001,
                 numThreads = -1)
  return np.transpose(out).todense()
  

def run_sc(X, num_features, iters, lamda):
  """Learn a Dictionary of features from input data X using Sparse Coding.

     Dictionary is initialized randomly. We learn the dictionary by looping
     over the following 2 steps:

     1. Fix dictionary and map X to sparse codes using coordinate-descent from
        INRIA SPAMS package

     2. Compute dictionary by solving Dict x S = X (or rather S' x Dict' = X')

     Parameters:
       X:     Input data as (num_patches, input_dim) matrix
       D:     Distonary of features
  """

  #Assuming X is num_patches x input_dim
  # initialize dictionary
  dict = np.random.randn(num_features, X.shape[1])
  dict = dict / np.sqrt((dict**2).sum(1) + 1e-20).reshape(dict.shape[0], 1)
  for itr in range(iters):
    print "Running sparse coding: iternation=",  itr
    t1 = time.time()
    S = sparse_codes(X, dict, lamda)
    print time.time() - t1
    (dict, _, _, _) = np.linalg.lstsq(S, X)
    dict = dict / np.sqrt((dict**2).sum(1) + 1e-20).reshape(dict.shape[0], 1)
  return dict


def build_dict():
  """Main routine for reading images, sampling patched off them and then
     learning a dictionary of features from those patches using sparse
     coding
  """
  images = scipy.io.loadmat("IMAGES.mat")['IMAGES'] # load images from disk 
  patches = sample_images(images, 8, 20000)
  patches = np.transpose(patches)
  #display_network(np.random.permutation(patches)[0:200, :], 14)
  W = run_sc(patches, 121, 450, 0.015)

  # Save dictionary in a file
  np.save(DICT_FILE, W)
  

def load_dict():
  """ Load dictionary from a save file
  """
  return np.load(DICT_FILE + '.npy')

