## Convolution Neural Network Main Module

import time
import collections
import scipy
import numpy as np
from sampleImages import *
from displayNetwork import *
from computeNumGrad import *
from utils import *
from wrappers import *
from aeUtils import *
from sparseAECost import *
#from minFuncSGD import *

def ae_init(ae_config):
  visible_size = ae_config.visible_size
  hidden_size = ae_config.hidden_size

  W1_size = hidden_size * visible_size
  W2_size = visible_size * hidden_size
  b1_size = hidden_size
  b2_size = visible_size

  params_array = zeros(W1_size + W2_size + b1_size + b2_size)
  (W1, W2, b1, b2) = ae_array_to_stack(params_array, ae_config)
  
  # we'll choose weights uniformly from the interval [-r, r]
  r  = np.sqrt(6) / np.sqrt(hidden_size+visible_size+1)
  filter_init_rand(W1, (-r, r))
  filter_init_rand(W2, (-r, r))
  return params_array

def ae_train():
  """ Main routine for training Auto-encoder parameters and testing them
  """
  #======================================================================
  # STEP 0: Initialize Parameters and Load Data

  AEConfig = collections.namedtuple('AEConfig', 
                                    ['visible_size',
                                     'hidden_size',
                                     'sparsity_param',
                                     'lamda',
                                     'beta'])

  ae_config = AEConfig(
    visible_size   = 64,     # number of input units 
    hidden_size    = 25,     # number of hidden units 
    sparsity_param = 0.01,   # desired average activation of the hidden units.
    lamda          = 0.0001, # weight decay parameter       
    beta           = 3,      # weight of sparsity penalty term       
  )

  patches = sample_images()
  patches = np.transpose(patches)
  display_network(np.random.permutation(patches)[0:200, :], 14)
  patches = np.transpose(patches)

  params_array = ae_init(ae_config)

  #======================================================================
  # STEP 1: Sanity check the gradient computation using numerical gradient
  # computation

  DEBUG = False  # set this to true to check gradient
  if DEBUG:

    db_config = AEConfig(
      visible_size   = 64,     # number of input units 
      hidden_size    = 3,     # number of hidden units 
      sparsity_param = 0.01,   # desired average activation of the hidden units.
      lamda          = 0.0001, # weight decay parameter       
      beta           = 3,      # weight of sparsity penalty term       
    )
    db_patches = patches[0:10000]
    db_params_array = ae_init(db_config)

    t1 = time.time()
    (cost, db_grad) = sparse_ae_cost(db_params_array, db_config, db_patches)
    print time.time() - t1

    num_grad = compute_numerical_gradient(
      lambda p: sparse_ae_cost(p, db_config, db_patches, True),
      db_params_array)

    # Use this to visually compare the gradients side by side
    for i in (range(num_grad.size)):
       print num_grad[i], db_grad[i]

    diff = np.linalg.norm(num_grad-db_grad)/np.linalg.norm(num_grad+db_grad)
    # should be small. in our implementation, these values are usually 
    # less than 1e-9.
    print diff
 
    assert diff < 1e-9, 'difference too large. check your gradient computation again'

  #======================================================================
  # STEP 2: Train the autoencoder

  t1 = time.time()
  #x = np.array(params_array, np.float64)
  (opttheta, cost, _) = scipy.optimize.fmin_l_bfgs_b(
    lambda p: sparse_ae_cost(p, ae_config, patches),
    params_array,
    maxiter = 1000, iprint = 0)
  print time.time() - t1

  (W1, _, _, _) = ae_array_to_stack(opttheta, ae_config)
  display_network(W1, 5)
  
  return (cost, opttheta)
