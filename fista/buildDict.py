import time
import spams
from scipy.sparse import csc_matrix
import numpy as np
from samplePatches import *
from displayNetwork import *

DICT_FILE = 'sparse_code_dict'

def sparse_codes(X, D, lamda):
  # set optimization paramters
  X0 = np.asfortranarray(np.transpose(X))
  D0 = np.asfortranarray(np.transpose(D))
  # Get a sparse matrix
  A0 = csc_matrix(np.zeros((D.shape[0], X.shape[0])))
  out = spams.cd(X0, D0, A0,
                 lambda1 = lamda, mode = 2, itermax = 1000, tol = 0.001,
                 numThreads = -1)
  return np.transpose(out).todense()
  

#Assuming X is num_patches x input_dim
def run_sc(X, num_features, iters, lamda):
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
  images = scipy.io.loadmat("IMAGES.mat")['IMAGES'] # load images from disk 
  patches = sample_images(images, 8, 20000)
  patches = np.transpose(patches)
  #display_network(np.random.permutation(patches)[0:200, :], 14)
  W = run_sc(patches, 121, 450, 0.015)
  np.save(DICT_FILE, W)
  

def load_dict():
  return np.load(DICT_FILE + '.npy')

