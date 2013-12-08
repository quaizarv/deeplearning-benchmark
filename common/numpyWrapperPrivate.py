################################################################################
#
# The following routines map between Fortran ordering of array dimension/indices
# to Numpy ordering
#
################################################################################

def numpy_dim(arr, dim):
  if (dim > 2):
    return len(arr.shape) - dim
  else:
    if (len(arr.shape) == 1):
      return 0;
    else:
      return len(arr.shape) + dim - 3

def numpy_dim_tuple (dim_tuple):
  if (type(dim_tuple) != tuple):
     dim_tuple = (dim_tuple,)
  if (len(dim_tuple) > 2):
    dim_list = list(dim_tuple)
    dim_list[0] = dim_tuple[1]
    dim_list[1] = dim_tuple[0]
    return tuple(dim_list[::-1])
  else:
    return dim_tuple

def numpy_index_tuple (arr, index_tuple):
  if (type(index_tuple) != tuple):
     index_tuple = (index_tuple,)
  if (len(arr.shape) > 2):
    idx_list = list(index_tuple)
    if (len(index_tuple) == len(arr.shape)):
      idx_list[0] = index_tuple[1]
      idx_list[1] = index_tuple[0]
    return tuple(idx_list[::-1])
  else:
    return index_tuple

def api_dim_tuple (dim_tuple):
  if (len(dim_tuple) > 2):
    dim_tuple = dim_tuple[::-1]
    dim_list = list(dim_tuple)
    dim_list[0] = dim_tuple[1]
    dim_list[1] = dim_tuple[0]
    return tuple(dim_list)
  else:
    return dim_tuple


