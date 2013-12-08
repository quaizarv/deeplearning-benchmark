## Convolution Neural Network Main Module

import time
import collections
import numpy
from loadMNISTImages import *
from loadMNISTLabels import *
from utils import *
from cnnCost import *
from computeNumGrad import *
from minFuncSGD import *

def cnn_init(cnn_config):
  # Initialize parameters for a single layer convolutional neural
  # network followed by a softmax layer.
  #                            
  # Parameters:
  #  cnn_config  -  configuration for a single CNN layer, e.g. various
  #                 dimensions
  # Returns:
  #  theta_tuple - tuple of parameter arrays with initialized weights
  #  theta_array - unrolled paramter arrays into a vector

  # Initialize parameters randomly based on layer sizes.

  num_images = cnn_config.num_images
  image_dim = cnn_config.image_dim
  num_filters = cnn_config.num_filters
  filter_dim = cnn_config.filter_dim
  pool_dim = cnn_config.pool_dim
  num_classes = cnn_config.num_classes

  assert (filter_dim < image_dim),'filterDim must be less that imageDim'

  # out_dim should be a multiple of pool_dim
  out_dim = (image_dim - filter_dim + 1)
  assert (out_dim % pool_dim) == 0, 'pool_dim must divide image_dim - filter_dim + 1'
  out_dim = out_dim/pool_dim
  hidden_size = out_dim*out_dim*num_filters

  Wc_size = filter_dim * filter_dim * num_filters
  Wd_size = num_classes * hidden_size
  bc_size = num_filters
  bd_size = num_classes
  
  params_array = zeros(Wc_size + Wd_size + bc_size + bd_size)
  (Wc, Wd, bc, bd) = array_to_stack(params_array, cnn_config)
  filter_init_randn(Wc);
  
  # we'll choose weights uniformly from the interval [-r, r]
  r  = np.sqrt(6) / np.sqrt(num_classes+hidden_size+1)
  filter_init_rand(Wd, (-r, r));

  theta_tuple = (Wc, Wd, bc, bd)
  return (theta_tuple, params_array)

def cnn_train():
  """ Main routine for training CNN parameters and testing them
  """
  #======================================================================
  # STEP 0: Initialize Parameters and Load Data

  # Configuration

  # Load MNIST Train
  images = load_MNIST_images('../../common/train-images-idx3-ubyte')
  labels = load_MNIST_labels('../../common/train-labels-idx1-ubyte')

  #images = images[0:10000]
  #labels = labels[0:10000]

  CNNConfig = collections.namedtuple('CNNConfig', 
                                     ['num_images',
                                      'image_dim',
                                      'num_filters',
                                      'filter_dim',
                                      'pool_dim',
                                      'num_classes'])

  cnn_config = CNNConfig(
    num_images  = 256, # minibatch size
    image_dim   = 28,
    num_filters = 20,  # Number of filters for conv layer
    filter_dim  = 9,   # Filter size for conv layer
    pool_dim    = 2,   # Pooling dimension
    num_classes = 10,  # Number of classes (MNIST images fall into 10 classes)
    )

  # Initialize Parameters
  (theta_tuple, _) = cnn_init(cnn_config)
                   
  #======================================================================
  # STEP 1: Gradient Check

  DEBUG = True;  # set this to true to check gradient
  if DEBUG:
    # To speed up gradient checking, we will use a reduced network and
    # a debugging data set
    debug_config = CNNConfig(
      num_images  = 10, # minibatch size
      image_dim   = 28,
      num_filters = 2,  # Number of filters for conv layer
      filter_dim  = 9,   # Filter size for conv layer
      pool_dim    = 5,   # Pooling dimension
      num_classes = 10,  # Number of classes (MNIST images fall into 10 classes)
      )

    db_images = zeros((debug_config.image_dim, debug_config.image_dim,
                       debug_config.num_images))
    db_labels = zeros(debug_config.num_images)
    for i in range(debug_config.num_images):
      db_images[i] = get_matrix(images, i)
      db_labels[i] = get_matrix(labels, i)

    (_, db_theta) = cnn_init(debug_config)

    # Check gradients
    num_grad = compute_numerical_gradient(
      lambda (x): cnn_cost_with_gradient(x, db_images, db_labels,
                                         debug_config, True),
      db_theta)

    (cost, db_grad) = cnn_cost_with_gradient(db_theta, db_images,
                                             db_labels, debug_config)
    
    # Use this to visually compare the gradients side by side
    for i in (range(num_grad.size)):
       print num_grad[i], db_grad[i]
    
    diff = np.linalg.norm(num_grad-db_grad)/np.linalg.norm(num_grad+db_grad)
    # should be small. in our implementation, these values are usually 
    # less than 1e-9.
    print diff
 
    assert diff < 1e-9, 'difference too large. check your gradient computation again'
  
    
  #======================================================================
  # step 2: learn parameters
                   
  sgdoptions = collections.namedtuple('sgdoptions', 
                                      ['epochs',
                                       'minibatch',
                                       'alpha',
                                       'momentum'])
  options = sgdoptions(
    epochs = 3,
    minibatch = 256,
    alpha = 1e-1,
    momentum = .95,
    )

  t1 = time.time()
  opt_theta_tuple = min_func_SGD(
    lambda a,b,c,d,e,f: cnn_cost(a, b, c, cnn_config, d, e, f),
    theta_tuple, images, labels, options)
  print time.time() - t1


  #======================================================================
  # step 3: test
  #  test the performance of the trained model using the mnist test set. your
  #  accuracy should be above 97# after 3 epochs of training

  test_images = load_MNIST_images('../common/t10k-images-idx3-ubyte')
  test_labels = load_MNIST_labels('../common/t10k-labels-idx1-ubyte')
  #test_images = test_images[0:1000]
  #test_labels = test_labels[0:1000]
  cnn_test_config = CNNConfig(
    num_images  = size(test_images, 0),
    image_dim   = 28,
    num_filters = 20,   # number of filters for conv layer
    filter_dim  = 9,    # filter size for conv layer
    pool_dim    = 2,    # pooling dimension
    num_classes = 10,   # number of classes (mnist images fall into 10 classes)
    )


  t2 = time.time()
  (cost, preds) = cnn_cost(opt_theta_tuple, test_images, test_labels,
                           cnn_test_config)
  print time.time() - t2

  #TBD fix this by using operations from the API
  acc = (np.ones(preds.size))[preds==test_labels].sum()/preds.size

  print "Accuracy is: ", acc, "\n"
  return (opt_theta_tuple, preds, test_labels)
