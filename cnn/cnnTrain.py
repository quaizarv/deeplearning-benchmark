## Convolution Neural Network Main Module

import time
import collections
import numpy
from loadMNISTImages import *
from loadMNISTLabels import *
from utils import *
from cnnUtils import *
from cnnCost import *
from computeNumGrad import *
from minFuncSGD import *

def cnn_init(cnn_config):
  """Initialize parameters for a single layer convolutional neural
     network followed by a softmax layer.
                              
     Parameters:
       cnn_config  -  configuration for a single CNN layer, e.g. various
                      dimensions
     Returns:
       theta_tuple - tuple of parameter arrays with initialized weights
       params_array - unrolled paramter arrays into a vector

     Initialize parameters randomly based on layer sizes.
  """

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

  # Initialize parameter randomly from the normal distribution for the convolve
  # layer
  filter_init_randn(Wc);
  
  # we'll choose weights uniformly from the interval [-r, r]
  r  = np.sqrt(6) / np.sqrt(num_classes+hidden_size+1)
  filter_init_rand(Wd, (-r, r));

  theta_tuple = (Wc, Wd, bc, bd)
  return (theta_tuple, params_array)

# Convert images and labels to the API implementation format
def convert_images_labels_to_api_format(images, labels):
  ishape = images.shape
  images = images.reshape(images.size)
  train_images = load_array(images, (ishape[1], ishape[2], ishape[0]))
  images.shape = ishape

  lshape = labels.shape
  labels = labels.reshape(labels.size)
  train_labels = load_array(labels, (lshape[0], ))
  labels.shape = lshape
  print train_labels.shape, train_images.shape
  return (train_images, train_labels)


def cnn_train():
  """ Main routine for training CNN parameters and testing them

     Returns: 

     opt_theta_tuple - a tuple of trained optimal parameters.
       Includes (Wc, Wd, bc, bd)  where:
       Wc - 3-dimensional tensor containing parameters for
            convolutional layer filters. The first dimension
            identifies the filter and the remaining 2 dimension
            are for the filter paramters
       Wd - parameter matrix for the softmax layer
       bc - intercept terms, one per filter at the conv layer
       bd - intercept terms, one per class in the softmax layer
       
  
     preds -  list of predictions for each test data instance 

  """
  #======================================================================
  # Initialize Parameters and Load Data
  #======================================================================

  # Configuration

  # Load MNIST Train
  images = load_MNIST_images('../mnist/train-images-idx3-ubyte')
  labels = load_MNIST_labels('../mnist/train-labels-idx1-ubyte')

  # Initialize configuration for the CNN layers

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

  # Convert images and labels to the API implementation format
  (train_images, train_labels) = convert_images_labels_to_api_format(
    images, labels)
  #train_images = images
  #train_labels = labels

  # Initialize Parameters
  (theta_tuple, _) = cnn_init(cnn_config)

                   
  #======================================================================
  # Code for sanity checking of the gradient computation
  #======================================================================

  DEBUG = False;  # set this to true to check gradient
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

    db_images = images[0:debug_config.num_images]
    db_labels = labels[0:debug_config.num_images]

    # Convert images and labels to the API implementation format
    (db_images, db_labels) = convert_images_labels_to_api_format(
      db_images, db_labels)

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
  # Train parameters
  #======================================================================

  # Stochastic Grandient configuration
    
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
    theta_tuple, train_images, train_labels, options)
  print time.time() - t1


  #======================================================================
  #  Test the performance of the trained model using the mnist test set.
  #  Accuracy should be above 97% after 3 epochs of training
  #======================================================================

  test_images = load_MNIST_images('../mnist/t10k-images-idx3-ubyte')
  test_labels = load_MNIST_labels('../mnist/t10k-labels-idx1-ubyte')

  # Convert images and labels to the API implementation format
  (test_images, test_labels) = convert_images_labels_to_api_format(
    test_images, test_labels)

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
  return (opt_theta_tuple, preds)
