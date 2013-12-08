import scipy
from scipy.signal import *
import numpy as np
from numpyWrapperPrivate import *

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
  filter[0:filter.size] = (0.1) * np.random.randn(filter.size)
  filter.shape = filter_shape

def filter_init_rand(filter, interval):
  filter_shape = filter.shape
  filter.shape = (filter.size, )
  (l, u) = interval
  filter[0:filter.size] = np.random.rand(filter.size) * (u - l) + l
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

def convolve_2d(A, B):
  return convolve2d(A, np.rot90(B, 2), "valid")

def convolve_T4(A, B):
  (d0, d1, d2) = A.shape
  R = np.zeros((d0, B.shape[0], d1-B.shape[2]+1, d2-B.shape[2]+1),
               dtype = floatX)
  for i in range(B.shape[0]):
    b = B[i]
    for j in range(A.shape[0]):
      a = A[j]
      R[j, i] = convolve2d(a, np.rot90(b, 2), "valid")
  return R

def average_pool_T4(A, pool_dim):
  b = np.ones((pool_dim, pool_dim), dtype = floatX);
  (d0, d1, d2, d3) = A.shape
  R = np.zeros((d0, d1, d2/pool_dim, d3/pool_dim), dtype = floatX)
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      a = A[i, j]
      result = convolve2d(a, b, "valid")
      # Subsample
      R[i,j] = result[0::pool_dim, 0::pool_dim] / (pool_dim * pool_dim)
  return R

def grad_conv_T4(A, B):
  out_dim = A.shape[2] - B.shape[2] + 1
  R = np.zeros((B.shape[1], out_dim, out_dim), dtype = floatX)
  for i in range(A.shape[0]):
    for f in range(B.shape[1]):
      R[f]  = R[f] + convolve2d(A[i], np.rot90(B[i, f], 2), "valid")
  return R

"""
def upsample(activations, pool_dim):
  return arrayfun(lambda elem: (1.0/(pool_dim * pool_dim)) * elem,
                  np.kron(activations, np.ones((pool_dim, pool_dim),
                                                dtype = floatX)));
"""

def upsample_T4(A, pd):
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A.reshape(A.size)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = scipy.linalg.kron(A,B).reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  return R

def soft_threshold(A):
  return 1/(1 + np.exp(-A))
  #return arrayfun(lambda elem: 1/(1 + np.exp(-elem)), A)

def filter_convolve(A, filter, intercept):
  shape = intercept.shape
  intercept.shape = (shape[0], 1, 1)
  R = bsxfun(lambda x, y: x + y, convolve_T4(A, filter), intercept)
  R = soft_threshold(R)
  intercept.shape = shape
  return R



################################################################################
#
# Following routines are currently USED by AE implementaion
#
# These are for generic neural net back propagation
#
################################################################################

def matrix_transpose_matrix_multiply(m1, m2):
  """ Perform (m1' x m2)
  """
  return np.dot(np.transpose(m1), m2)

def matrix_matrix_transpose_multiply(m1, m2):
  """ Perform (m1 x m2')
  """
  return np.dot(m1, np.transpose(m2))

def filter_multiply(filter, intercept, data_matrix):
  """ Perform (filter x data_matrix + intercept)
  """
  return np.dot(filter, data_matrix) + intercept
