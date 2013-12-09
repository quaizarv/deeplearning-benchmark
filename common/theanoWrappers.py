import scipy
from scipy.signal import *
import numpy
from numpyWrapperPrivate import *
import theano
import theano.tensor as T
from theano.tensor.signal import conv
from theano import sandbox, shared, Out
import theano.sandbox.neighbours as TSN

floatX = 'float64'


# Note on Array Index order: This API assumes Fortran order, i.e.  (1, 2, 3)
# where leftmost index is packed closest in memory, then 2, then 3 and so on


################################################################################
#
# The following routines allow creating, initializing arrays, reshaping them,
# extracting/setting matrices from/to multi-dim arrays etc.
#
################################################################################

def size(arr, dim = -1):
  if (dim == -1):
    return api_dim_tuple(arr.shape)
  return arr.shape[numpy_dim(arr, dim)]

# Array creation
def zeros(dim_tuple):
  """ create an array of shape given by dim_tuple

  Parameters:
    dim_tuple : a tuple specifying the dimensions of the array

  Returns:
    array     : the created array
  """
  return np.zeros(numpy_dim_tuple(dim_tuple), dtype = floatX)

def get_matrix(arr, index_tuple):
  """Extract a matrix from a array of matrices
  
     If the array has dimensions (d1, d2, d3, d4, d5) - then the matrix
     corresponds to first 2 dimensions, i.e. (d1, d2) and is located at index
     (:, :, d3, d4, d5). Pass in (d3, d4, d5) as the index_tuple to locate the
     matrix

     Parameters:
       index_tuple : identifies the matrix position in the multi-dimensional
                     array
  """
  return arr[numpy_index_tuple(arr, index_tuple)]

def set_matrix(arr, index_tuple, matrix):
  """Save a matrix to an array of matrices
  
     If the array has dimensions (d1, d2, d3, d4, d5) - then the matrix occupies
     the space corresponding to last 2 dimensions, i.e. (d4, d5) and is located
     at index (d1, d2, d3). Pass in (d1, d2, d3) as the index_tuple to locate
     the matrix

     Parameters:
       index_tuple : identifies the matrix position in the multi-dimensional
                     array
  """
  arr[numpy_index_tuple(arr, index_tuple)] = matrix

def reshape(arr, dim_tuple):
  """Reshape the array as specified by the passed dimension tuple 
  
  Parameters:
    dim_tuple - tuple specifying the new dimension for the array
  """
  arr.shape = numpy_dim_tuple(dim_tuple)

def get_array_view(array1d, start, dim_tuple):
  dim_tuple = numpy_dim_tuple(dim_tuple)
  end = start + reduce(lambda sz, dim: sz * dim, dim_tuple)
  result_array = array1d[start:end]
  result_array.shape = dim_tuple
  return (result_array, end)

def filter_init_randn(filter):
  filter_shape = filter.shape
  filter.shape = (filter.size, )
  filter[0:filter.size] = (0.1) * np.random.randn(filter.size).astype(floatX)
  filter.shape = filter_shape

def filter_init_rand(filter, interval):
  filter_shape = filter.shape
  filter.shape = (filter.size, )
  (l, u) = interval
  filter[0:filter.size] = np.random.rand(filter.size).astype(floatX) * (u - l) + l
  filter.shape = filter_shape

def extract_random_minibatch(dst_data, dst_labels, src_data, src_labels, rp):
  for i in (range(size(rp, 1))):
    set_matrix(dst_data, i, get_matrix(src_data, rp[i]))
    dst_labels[i] = src_labels[rp[i]]
    #dst_data[i,:,:] = src_data[rp[i], :, :]

################################################################################
#
# The following routines allow binary-singleton expansion operations,
# elementwise and reduction operations on multi-dim arrays
#
################################################################################

def bsxfun(f, A, B):
  return f(A, B)

def arrayfun(f, A):
  return f(A)

def reducefun(f, A, dim):
  return f.reduce(A, numpy_dim(A, dim))

################################################################################
#
# Multidimensional matrix multiplications
#
################################################################################

def matrix_matrix4D_multiply(A, B):
  # Reshape activations into 2-d matrix with the last dimension as column
  # dimension of the matrix

  # In numpy we have dimensions reversed
  orig_shape = B.shape
  B.shape = (B.shape[0], B.size/B.shape[0])
  B_transpose = np.transpose(B)
  B.shape = orig_shape

  return np.dot(A, B_transpose)

def matrix_matrix4D_transpose_multiply(A, B):

  # Given numpy's ordering of array dimensions reshaping B gives us the
  # transpose
  orig_shape = B.shape
  B.shape = (B.shape[0], B.size/B.shape[0])
  result =  np.dot(A, B)
  B.shape = orig_shape
  return result

def matrix_matrix_transpose_multiply_to_4D(A, B, dim_tuple):
  """ Perform (filter' x data_matrix)

      Perform a matrix multiplication between transpose of the 'filter' matrix
      and the 'data_matrix'
    """
  result = np.dot(np.transpose(A), B)

  # Deal with numpy's 3-2-1 indexing - if A is (M,K) and B is (K,N) then result
  # is (M,N) - now we are trying to decompose M into 3-dimensions as (M1, M2,
  # M3) so that result is (M1, M2, M3, N). But this is expressed as shape (N,
  # M1, M2, M3) in numpy. To get to this shape, we have to first transpose
  # result to (N, M) and then reshape to get to (N, M1, M2, M3)
  result = np.transpose(result)

  reshape(result, dim_tuple)
  return result

################################################################################
#
# miscellaneous routines
#
################################################################################

def bit_numbers_to_bit_vectors(bit_count, bit_num_array):
  bit_matrix = np.zeros((bit_count, bit_num_array.shape[0]), dtype = floatX)
  for i in range(bit_num_array.shape[0]):
    bit_matrix[bit_num_array[i], i] = 1
  return bit_matrix

def matrix_dot_product(A, B):
  return (A * B).sum()

def argmax_by_column(A):
  return np.argmax(A, 0)

################################################################################
#
# CNN specific routines which have significant performance impact
#
################################################################################


def rot180_T4(A):
  Ashape = A.shape
  if (len(A.shape) < 4):
    A.shape = (Ashape[0], 1, Ashape[1], Ashape[2])
  R = np.zeros(A.shape, dtype = floatX)
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      R[i, j] = np.fliplr(np.flipud(A[i,j]))
  A.shape = Ashape
  R.shape = Ashape
  return R;

floatX1 = 'float32'
A = T.tensor4('conv_fun_A', dtype = floatX1)
B = T.tensor4('conv_fun_B', dtype = floatX1)
conv_out = T.nnet.conv2d(A, B)
conv_fun = theano.function([A, B], conv_out, name = 'conv_fun')

z = T.tensor4('z', dtype = floatX1)
soft_th_out = 1 / (1 + T.exp(-z))
soft_th_fun = theano.function([z], soft_th_out, name = 'soft_th_fun')

def soft_threshold(z):
  zshape = z.shape
  if (len(z.shape) == 2):
    z.shape = (1, 1, zshape[0], zshape[1])
  R = soft_th_fun(z)
  R.shape = zshape
  z.shape = zshape
  return R

InterceptBroadcaster = T.TensorType(dtype = floatX1,
                                    broadcastable = [True, False, True, True])
fc_W = T.tensor4('filter_convolve_weight_matrix', dtype = floatX1)
fc_b = InterceptBroadcaster('filter_convolve_intercept')
                            
fc_z = conv_out + fc_b
fc_fun = theano.function([A, fc_W, fc_b], soft_th_out,
                         givens = [(B, fc_W), (z, fc_z)],
                         name = 'filter_convolve_function')

def filter_convolve(A, filter, intercept):
  Ishape = intercept.shape
  intercept.shape = (1, Ishape[0], 1, 1)
  Ashape = A.shape
  A.shape = (Ashape[0], 1, Ashape[1], Ashape[2])
  Bshape = filter.shape
  filter.shape = (Bshape[0], 1, Bshape[1], Bshape[2])
  R = fc_fun(A.astype(floatX1), rot180_T4(filter).astype(floatX1),
             intercept.astype(floatX1))
  A.shape = Ashape
  filter.shape = Bshape
  intercept.shape = Ishape
  return R

pdim = T.scalar('pool dim', dtype = floatX1)
pool_inp = T.tensor4('pool input', dtype = floatX1)
pool_sum = TSN.images2neibs(pool_inp, (pdim, pdim))
pool_out = pool_sum.mean(axis=-1) 
pool_fun = theano.function([pool_inp, pdim], pool_out, name = 'pool_fun')
def average_pool_T4(A, pool_dim):
  # Warning: pool_fun returns a 1-D vector, we need to reshape it into a 4-D
  # tensor
  temp = pool_fun(A, pool_dim)
  temp.shape = (A.shape[0], A.shape[1], A.shape[2]/pool_dim,
                A.shape[3]/pool_dim)
  return temp


def convolve_T4(A, B):
  Ashape = A.shape
  if (len(A.shape) < 4):
    A.shape = (Ashape[0], 1, Ashape[1], Ashape[2])
  Bshape = B.shape
  if (len(B.shape) < 4):
    B.shape = (Bshape[0], 1, Bshape[1], Bshape[2])
  R = conv_fun(A.astype(floatX1), rot180_T4(B).astype(floatX1))
  A.shape = Ashape
  B.shape = Bshape
  return R

def grad_conv_T4(A, B):
  Ashape = A.shape
  if (len(A.shape) < 4):
    A.shape = (1, Ashape[0], Ashape[1], Ashape[2])
  B = B.swapaxes(0, 1)
  R = convolve_T4(A, B)
  A.shape = Ashape
  return R

def upsample(activations, pool_dim):
  ups = np.kron(activations, np.ones((pool_dim, pool_dim),
                                     dtype = floatX));
  return ups * (1.0/(pool_dim * pool_dim))
  #return arrayfun(lambda elem: (1.0/(pool_dim * pool_dim)) * elem,

from theano.sandbox.linalg import kron
upsample_pd = T.scalar("upsample_pool_dim", dtype = floatX1)
upsample_inp = T.vector("upsample_input", dtype = floatX1)
upsample_f = T.matrix("upsample_filter", dtype = floatX1)
kron_out = kron(upsample_inp, upsample_f) * (1.0/ (upsample_pd ** 2))
kron_fun = theano.function([upsample_inp, upsample_f, upsample_pd], kron_out,
                           name = 'kron_fun')

def upsample_T4(A, pd):
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A = A.reshape(A.size)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = kron_fun(A.astype(floatX1), B.astype(floatX1), pd)
  R = R.reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  #R = R * (1.0/(pd * pd))
  return R

def upsample_T4_old(A, pd):
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A = A.reshape(A.size)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = scipy.linalg.kron(A,B).reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  return R


################################################################################
#
# Following routines are currently USED by AE implementaion
#
# These are for generic neural net back propagation
#
################################################################################

InterceptBroadcaster2D = T.TensorType(dtype = floatX1,
                                      broadcastable = [False, True])
f_W = T.matrix("f_w", dtype = floatX1)
data_mat = T.matrix("data_mat", dtype = floatX1)
f_b = InterceptBroadcaster2D('f_b')
#f_b = T.matrix("f_b", dtype = floatX1)
fm_out = T.dot(f_W, data_mat) + f_b
fm_fun = theano.function([f_W, f_b, data_mat], fm_out,
                                  name = 'filter_multiply')
def filter_multiply(filter, intercept, data_matrix):
  return fm_fun(filter.astype(floatX1), 
                intercept.astype(floatX1),
                data_matrix.astype(floatX1))

m1 = T.matrix("m1", dtype = floatX1)
m2 = T.matrix("m2", dtype = floatX1)
m1_transp = m1.dimshuffle(1, 0)
m2_transp = m2.dimshuffle(1, 0)
mmtm_out = T.dot(m1, m2_transp)
mmtm_fun = theano.function([m1, m2], mmtm_out, name = 'mat_mat_trans_multiply')

def matrix_matrix_transpose_multiply(m1, m2):
  return mmtm_fun(m1.astype(floatX1), m2.astype(floatX1))

mtmm_out = T.dot(m1_transp, m2)
mtmm_fun = theano.function([m1, m2], mtmm_out, name = 'mat_trans_mat_multiply')

def matrix_transpose_matrix_multiply(m1, m2):
  return mtmm_fun(m1.astype(floatX1), m2.astype(floatX1))


################################################################################
#
# Following routines are currently NOT USED by CNN implementaion
#
# In case we move back to computing Wc gradient, by doing per-(image, filter)
# convolve
#
################################################################################

A = T.tensor4('conv_fun_A', dtype = floatX)
B = T.tensor4('conv_fun_B', dtype = floatX)
conv_out2 = T.nnet.conv2d(A, B)
conv_fun2 = theano.function([A, B], conv_out2, name = 'conv_fun2')

def convolve_T2(A, B):
  A.shape = (1, 1, A.shape[0], A.shape[1])
  B.shape = (1, 1, B.shape[0], B.shape[1])
  C = conv_fun2(A, rot180_T4(B))
  A.shape = (A.shape[2], A.shape[3])
  B.shape = (B.shape[2], B.shape[3])
  C.shape = (C.shape[2], C.shape[3])
  return C

# This will be needed in case we break filter_convolve into convolve, 
# intercept add and threshold functions
"""
def filter_convolve(A, filter, intercept):
  shape = intercept.shape
  intercept.shape = (shape[0], 1, 1)
  R = bsxfun(lambda x, y: x + y, convolve_T4(A, filter), intercept)
  intercept.shape = shape
  return R
"""


