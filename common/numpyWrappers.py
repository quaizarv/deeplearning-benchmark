import scipy
from scipy.signal import *
import numpy as np
from arrayIndexMapping import *

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
  """Get size of an array either in a particular dimension or for all
  dimensions

    Parameters:
     arr: array whose size needs to be extracted

     dim: a particual dimension (axis) or -1 if the size of the entire array
          is required
  """
  if (dim == -1):
    return api_dim_tuple(arr.shape)
  return arr.shape[numpy_dim(arr, dim)]

# Array creation
def zeros(dim_tuple):
  """ create an array of shape given by dim_tuple and initialize to to 0s

  Parameters:
    dim_tuple : a tuple specifying the dimensions of the array

  Returns:
    array     : the created array
  """
  return np.zeros(numpy_dim_tuple(dim_tuple), dtype = floatX)

def load_array(np_array1d, fortran_dim_tuple):
  """API definition: 
         copy data from a 1-dimensional NUMPY array to a multi-dimensional
     array in the API implementation specific format

     Parameters:
      np_array1d: 1-dimenstional array in numpy format
      dim_tuple: array dimensions in Fortran order

     Return:
      Multi-dimensional array with shape as 'dim_tuple'
  """

  """
     API implementation in NUMPY: 
  """
  api_data = np.zeros(np_array1d.size)
  api_data[:] = np_array1d[:]
  api_data.shape = numpy_dim_tuple(fortran_dim_tuple)
  return api_data

def reshape(arr, dim_tuple):
  """Reshape the array as specified by the passed dimension tuple 
  
  Parameters:
    dim_tuple - tuple specifying the new dimension for the array
  """
  arr.shape = numpy_dim_tuple(dim_tuple)

def get_array_view(array1d, start, dim_tuple):
  """ Extract a multi-dimensional array from a large flat 1-dimenstional array
  
      Parameters:
       array1d: large 1-dim array
       start: start of the multidimensional array in array1d
       dim_tuple: shape of the output array

      Return:
        multi-dimensional array 
  """
  dim_tuple = numpy_dim_tuple(dim_tuple)
  end = start + reduce(lambda sz, dim: sz * dim, dim_tuple)
  result_array = array1d[start:end]
  result_array.shape = dim_tuple
  return (result_array, end)

def filter_init_randn(filter):
  """ Initial a filter array randomly from a normal distribution
  """
  filter_shape = filter.shape
  filter.shape = (filter.size, )
  filter[0:filter.size] = (0.1) * np.random.randn(filter.size)
  filter.shape = filter_shape

def filter_init_rand(filter, interval):
  """ Initial a filter array randomly from a unifrom distribution given
      by the interval tuple
  """
  filter_shape = filter.shape
  filter.shape = (filter.size, )
  (l, u) = interval
  filter[0:filter.size] = np.random.rand(filter.size) * (u - l) + l
  filter.shape = filter_shape

def extract_random_minibatch(dst_data, dst_labels, src_data, src_labels, rp):
  """ Extract a minibatch of data using a random permutation of indices
  """
  for i in (range(size(rp, 1))):
    dst_data[i] = src_data[rp[i]]
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
  """Multiply 2 dimensional matrix A with a 4-dimensional tensor B - this
    is equivalent of flattening out the last 2-dimensions of B 
    and then multiplying A to the resulting matrix of B 
  """

  # In numpy we have dimensions reversed
  orig_shape = B.shape
  B.shape = (B.shape[0], B.size/B.shape[0])
  B_transpose = np.transpose(B)
  B.shape = orig_shape

  return np.dot(A, B_transpose)

def matrix_matrix4D_transpose_multiply(A, B):
  """Multiply 2 dimensional matrix A with the transpose of a 4-dimensional
    tensor B - this is equivalent of flattening out the last 2-dimensions of B
    and then multiplying A to the transpose of the resulting matrix of B
  """

  # Given numpy's ordering of array dimensions reshaping B gives us the
  # transpose
  orig_shape = B.shape
  B.shape = (B.shape[0], B.size/B.shape[0])
  result =  np.dot(A, B)
  B.shape = orig_shape
  return result

def matrix_matrix_transpose_multiply_to_4D(A, B, dim_tuple):
  """ Multiply 2 matrices and reshape the result into a 4-dimenstional tensor.
      "dim_tuple" specifies the shape of the resulting tensor.
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

def matrix_elemwise_multiply(A, B):
  return (A * B).sum()

def argmax_by_column(A):
  return np.argmax(A, 0)

################################################################################
#
# CNN specific routines which have significant performance impact
#
################################################################################

def convolve_2d(A, B):
  #NUMPY specific Building block for conolve_T4 in
  return convolve2d(A, np.rot90(B, 2), "valid")

def convolve_T4(A, B):
  """ Convolve 4-dimensional tensor using 4-dimensional tensor B
  """
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
  """ Compute average pooling for a 4-dimensional tensor - this is equivalent
      to pooling over all the matrices stored in the 4-dim tensor
  """
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
  """ Convolve filter gradient in B with input data in A. B is a 4 dimensional
      array while A is 3-dimensional. TBD - this needs a clearer description
  """
  out_dim = A.shape[2] - B.shape[2] + 1
  R = np.zeros((B.shape[1], out_dim, out_dim), dtype = floatX)
  for i in range(A.shape[0]):
    for f in range(B.shape[1]):
      R[f]  = R[f] + convolve2d(A[i], np.rot90(B[i, f], 2), "valid")
  return R

def upsample_T4(A, pd):
  """Perform the reverse of average pooling on a 4-dimensioanl tensor by
      replicating an element in A "pd" times in the last 2 dimensions
      and then averaging
  """
  B = np.ones((1, pd*pd))
  Ashape = A.shape
  A.reshape(A.size)
  l = Ashape[0] * Ashape[1] * Ashape[2]
  R = scipy.linalg.kron(A,B).reshape(l,Ashape[3],pd,pd).swapaxes(1, 2).reshape(Ashape[0], Ashape[1], Ashape[2] * pd, Ashape[3] * pd)
  A.shape = Ashape
  return R

def soft_threshold(A):
  """ Compute the sigmoid function on A
  """
  return 1/(1 + np.exp(-A))

def filter_convolve(A, filter, intercept):
  """Perform convolution of A with filter followed by addition of an
  interncept term and then computing sigmoid function on the resulting 
  tensor
  """
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
