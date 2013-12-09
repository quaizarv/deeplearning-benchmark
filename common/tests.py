## Convolution Neural Network Main Module

import collections
import time
import scipy
import numpy as np
from loadMNISTImages import *
from loadMNISTLabels import *
from utils import *
import theano
import theano.tensor as T
from scipy.signal import *
from theano.tensor.signal import conv
import theano.sandbox.neighbours as TSN

floatX1 = 'float32'
W = T.tensor4('W', dtype = floatX1)
inp = T.tensor4('conv_op_input', dtype = floatX1)
conv_out = T.nnet.conv2d(inp, W)
conv_fun = theano.function([inp, W], conv_out)

def flip2d(A):
  shape = A.shape
  if (len(A.shape) < 4):
    A.shape = (shape[0], 1, shape[1], shape[2])
  R = np.zeros(A.shape)
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      R[i, j] = np.fliplr(np.flipud(A[i,j]))
  A.shape = shape
  return R;

def convolveTH4(images, W):
  Wshape = W.shape
  if (len(W.shape) < 4):
    W.shape = (W.shape[0], 1, W.shape[1], W.shape[2])
  Ishape = images.shape
  if (len(images.shape) < 4):
    images.shape = (images.shape[0], 1, images.shape[1], images.shape[2])
  R = conv_fun(images.astype(floatX1), flip2d(W).astype(floatX1))
  W.shape = Wshape
  images.shape = Ishape
  return R

def convolveTH4_1(images, W):
  Ishape = images.shape
  if (len(images.shape) < 4):
    images.shape = (images.shape[0], 1, images.shape[1], images.shape[2])
  Wshape = W.shape
  if (len(W.shape) < 4):
    W.shape = (W.shape[0], 1, W.shape[1], W.shape[2])
  tempI = shared(images)
  tempW = shared(W)
  c_out = T.nnet.conv2d(tempI, tempW)
  c_fun = theano.function([], c_out)
  R = c_fun()
  W.shape = Wshape
  images.shape = Ishape
  return R

A = T.dtensor3('W')
B = T.dtensor3('conv_op_input')
co = conv.conv2d(A, B)
conv_fun_T3 = theano.function([A, B], co)


def rot(A):
  return np.rot90(A, 2)

def average_pool(A, pool_dim):
  B = np.ones((1, pool_dim, pool_dim)).astype(floatX)
  R = np.zeros((A.shape[0], A.shape[1], A.shape[2]/pool_dim,
                A.shape[3]/pool_dim)).astype(floatX)
  for i in range(A.shape[0]):
    temp = convolveTH4(A[i], B)[:, 0, 0::pool_dim, 0::pool_dim]
    R[i] = temp / (pool_dim * pool_dim)
  return R

pdim = T.scalar('pool dim', dtype = floatX)
pool_inp = T.tensor4('pool input', dtype = floatX)
pool_sum = TSN.images2neibs(pool_inp, (pdim, pdim))
pool_out = pool_sum.mean(axis=-1) 
pool_fun = theano.function([pool_inp, pdim], pool_out)

def pool_th(A, pool_dim):
  temp = pool_fun(A, pool_dim)
  temp.shape = (A.shape[0], A.shape[1], A.shape[2]/pool_dim,
                A.shape[3]/pool_dim)
  return temp

def convolveTH2(A, B):
  A.shape = (1, 1, A.shape[0], A.shape[1])
  B.shape = (1, 1, B.shape[0], B.shape[1])
  C = conv_fun(A, B)
  A.shape = (A.shape[2], A.shape[3])
  B.shape = (B.shape[2], B.shape[3])
  C.shape = (C.shape[2], C.shape[3])
  return C

def convolveTH23(A, B):
  A.shape = (1, A.shape[0], A.shape[1])
  B.shape = (1, B.shape[0], B.shape[1])
  C = conv_fun_T3(A, B)
  A.shape = (A.shape[1], A.shape[2])
  B.shape = (B.shape[1], B.shape[2])
  C.shape = (C.shape[2], C.shape[3])
  return C


def convolveTH3(images, W):
  R = conv_fun_T3(images, W)
  return R

def loop_convolve_th(A, B, R):
  for i in range(B.shape[0]):
    b = B[i]
    for j in range(A.shape[0]):
      a = A[j]
      R[j, i] = convolveTH2(a, rot(b))
  return R

def convolveT4(A, B, R):
  for i in range(B.shape[0]):
    b = B[i]
    for j in range(A.shape[0]):
      a = A[j]
      R[j, i] = convolve2d(a, rot(b), "valid")
  return R

def rot2d(A):
  shape = A.shape
  A.shape = (shape[0], 1, shape[1], shape[2])
  R = np.zeros(A.shape)
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      R[i, j] = np.rot90(A[i,j], 2)
  A.shape = shape
  return R;

def filter_init_randn(shape):
  return (0.1) * np.random.randn(*numpy_dim_tuple(shape)).astype(floatX)

def filter_init_rand(shape, interval):
  (l, u) = interval
  temp = np.random.rand(*numpy_dim_tuple(shape)) * (u - l) + l
  return temp.astype(floatX)

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

  #params_array = zeros(Wc_size + Wd_size + bc_size + bd_size)
  #(Wc, Wd, bc, bd) = array_to_stack(params_array, cnn_config)
  Wc = filter_init_randn((filter_dim, filter_dim, num_filters))
  
  # we'll choose weights uniformly from the interval [-r, r]
  r  = np.sqrt(6) / np.sqrt(num_classes+hidden_size+1)
  Wd = filter_init_rand((num_classes, hidden_size), (-r, r))

  bc = np.zeros(num_filters)
  bd = np.zeros(num_classes)

  theta_tuple = (Wc, Wd, bc, bd)
  return theta_tuple


def grad_conv(A, B):
  out_dim = A.shape[2] - B.shape[2] + 1
  R = np.zeros((B.shape[1], out_dim, out_dim))
  for i in range(A.shape[0]):
    for f in range(B.shape[1]):
      R[f]  = R[f] + convolve2d(A[i], rot(B[i, f]), "valid")
  return R
               
def grad_conv_T4(A, B):
  Ashape = A.shape
  A.shape = (1, Ashape[0], Ashape[1], Ashape[2])
  R = convolveTH4(A, flip2d(np.swapaxes(B, 0, 1)))
  A.shape = Ashape
  return R

def grad_conv_T4_TH2(A, B):
  out_dim = A.shape[2] - B.shape[2] + 1
  R = np.zeros((B.shape[1], out_dim, out_dim))
  for i in range(A.shape[0]):
    for f in range(B.shape[1]):
      R[f]  = R[f] + convolveTH2(A[i], rot(B[i, f]))
  return R

def grad_conv_T4_TH3(A, B):
  out_dim = A.shape[2] - B.shape[2] + 1
  R = np.zeros((B.shape[1], out_dim, out_dim))
  for i in range(A.shape[0]):
    for f in range(B.shape[1]):
      R[f]  = R[f] + convolveTH23(A[i], rot(B[i, f]))
  return R

from theano.sandbox.linalg import kron
m1 = T.tensor4("kron_input_1")
m2 = T.tensor4("kron_input_2")
kron_out = kron(m1, m2)
kron_fun = theano.function([m1, m2], kron_out)

def upsample_TH(A, pool_dim):
  B = np.ones((pool_dim, pool_dim))
  R = np.zeros((A.shape[0], A.shape[1], A.shape[2] * pool_dim,
                A.shape[3] * pool_dim))
  for i in range(R.shape[0]):
    for j in range(R.shape[1]):
      R[i, j] = kron_fun(A[i, j], B)
  R = R / (pool_dim * pool_dim)
  return R

def upsample_NP(A, pool_dim):
  B = np.ones((pool_dim, pool_dim))
  R = np.zeros((A.shape[0], A.shape[1], A.shape[2] * pool_dim,
                A.shape[3] * pool_dim))
  for i in range(R.shape[0]):
    for j in range(R.shape[1]):
      R[i, j] = np.kron(A[i, j], B) 
  R = R / (pool_dim * pool_dim)
  return R

  R = R / (pool_dim * pool_dim)


from theano.sandbox.linalg import kron
m1 = T.matrix("m1", dtype = floatX1)
m2 = T.matrix("m2", dtype = floatX1)
mmult_out = T.dot(m1, m2)
dot = theano.function([m1, m2], mmult_out)

f_W = T.matrix("f_w", dtype = floatX1)
data_mat = T.matrix("data_mat", dtype = floatX1)

InterceptBroadcaster2D = T.TensorType(dtype = floatX1,
                                      broadcastable = [False, True])
f_b = InterceptBroadcaster2D('f_b')
#f_b = T.matrix("f_b", dtype = floatX1)
fm_out = T.dot(f_W, data_mat) + f_b
filter_multiply = theano.function([f_W, f_b, data_mat], fm_out,
                                  name = 'filter_multiply')
m1_transp = m1.dimshuffle(1, 0)
m2_transp = m2.dimshuffle(1, 0)
mmtm_out = T.dot(m1, m2_transp)
matrix_matrix_transpose_multiply = theano.function([m1, m2], mmtm_out)

mtmm_out = T.dot(m1_transp, m2)
matrix_transpose_matrix_multiply = theano.function([m1, m2], mtmm_out)


def mmult_test():
  A = np.random.random((2000, 10))
  B = np.random.random((10, 256))
  t = time.time()
  for i in range(50):
    R1 = dot(A, B)
  print time.time() - t
  t = time.time()
  for i in range(50):
    R2 = np.dot(A, B)
  print time.time() - t
  return (R1, R2)

A = T.tensor4("A", dtype = floatX)

s = 1 / (1 + T.exp(-A))

soft_th = theano.function([A], s)

def soft_np(A):
  return 1/(1 + np.exp(-A))

def cnn_test():
  """ Main routine for training CNN parameters and testing them
  """
  #======================================================================
  # STEP 0: Initialize Parameters and Load Data

  # Configuration

  # Load MNIST Train
  images = load_MNIST_images('../common/train-images-idx3-ubyte')
  labels = load_MNIST_labels('../common/train-labels-idx1-ubyte')

  CNNConfig = collections.namedtuple('CNNConfig', 
                                     ['num_images',
                                      'image_dim',
                                      'num_filters',
                                      'filter_dim',
                                      'pool_dim',
                                      'num_classes'])

  cnn_config = CNNConfig(
    num_images  = 256,
    image_dim   = 28,
    num_filters = 20,  # Number of filters for conv layer
    filter_dim  = 9,   # Filter size for conv layer
    pool_dim    = 2,   # Pooling dimension
    num_classes = 10,  # Number of classes (MNIST images fall into 10 classes)
    )

  # Initialize Parameters
  (Wc, Wd, bc, bd) = cnn_init(cnn_config)

  #(d0, d1, d2) = images.shape
  #act1 = zeros((256, Wc.shape[0], d1-Wc.shape[2]+1, d2-Wc.shape[2]+1))

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    a1 = convolveTH4(batch, Wc)
    #l1 = soft_th(a1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #a2 = convolveTH4_1(batch, Wc)
    #l1 = soft_th(a1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    a3 = pool_th(a1, 2)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    a4 = average_pool(a1, 2)
  print time.time() - t


  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #a1 = convolveTH4(batch, Wc)
    #l2 = soft_np(a1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #a1 = convolveTH4(batch, Wc)
    #a1 = loop_convolve_th(batch, Wc, act1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #g1 = grad_conv_T4(batch, a1)
    #a4 = loop_convolve_th(batch, Wc, act1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #g2 = grad_conv_T4_TH2(batch, a1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #g3 = grad_conv_T4_TH3(batch, a1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #g4 = grad_conv(batch, a1)
    #a3 = loop_convolve_th(batch, Wc, act1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #a2 = convolveTH3(batch, Wc)
    #a2 = loop_convolve_th(batch, Wc, act1)
  print time.time() - t

  t = time.time()
  for i in range(23):
    batch = images[i*256:(i+1)*256]
    #a5 = flip2d(batch)
    #a5 = convolveT4(batch, Wc, act1)
  print time.time() - t

  #Wc.shape = (Wc.shape[0], 1, Wc.shape[1], Wc.shape[2])
  #batch.shape = (batch.shape[0], 1, batch.shape[1], batch.shape[2])
  #R2 = conv_fun(batch, Wc)

  return (a1, a3, a4)


from theano.sandbox.linalg import kron
upsample_inp = T.vector("upsample_input", dtype = floatX)
upsample_f = T.matrix("upsample_filter", dtype = floatX)
kron_out = kron(upsample_inp, upsample_f)
kron_fun = theano.function([upsample_inp, upsample_f], kron_out)

def upsample_T4(A, pd):
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A = A.reshape(A.size)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = kron_fun(A, B)
  R = R.reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  return R

def upsample_T4_old(A, pd):
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A.shape = (A.size, 1)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = scipy.linalg.kron(A,B).reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  return R
