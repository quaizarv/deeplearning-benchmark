from wrappers import *

def array_to_stack(param_stack, cnn_config):
  num_images = cnn_config.num_images
  image_dim = cnn_config.image_dim
  num_filters = cnn_config.num_filters
  filter_dim = cnn_config.filter_dim
  pool_dim = cnn_config.pool_dim
  num_classes = cnn_config.num_classes

  out_dim = (image_dim - filter_dim + 1)/pool_dim
  hidden_size = out_dim*out_dim*num_filters

  start = 0
  (Wc, start) = get_array_view(param_stack, start, 
                               (filter_dim, filter_dim, num_filters))
  (Wd, start) = get_array_view(param_stack, start, (num_classes, hidden_size))
  (bc, start) = get_array_view(param_stack, start, (num_filters, 1))
  (bd, start) = get_array_view(param_stack, start, (num_classes, 1))
  return (Wc, Wd, bc, bd)


def scalar_multiply(scalar_val, A):
  return arrayfun(lambda elem: scalar_val * elem, A)

def scalar_minus(scalar_val,  A):
  return arrayfun(lambda elem: scalar_val - elem, A)

def scalar_plus(scalar_val,  A):
  return arrayfun(lambda elem: scalar_val + elem, A)

def plus(A, B):
  return bsxfun(lambda x, y: x + y, A, B)

def minus(A, B):
  return bsxfun(lambda x, y: x - y, A, B)

def multiply(A, B):
  return bsxfun(lambda x, y: x * y, A, B)

def sum_by_column(A):
  return reducefun(np.add, A, 1)

def sum_by_row(A):
  return reducefun(np.add, A, 2)

def sigmoid_gradient(A):
  return arrayfun(lambda elem: elem * (1 - elem), A)

def softmax(A):
  h = arrayfun(lambda elem: np.exp(elem), A)
  return bsxfun(lambda x, y: x / y, h, sum_by_column(h))

def softmax_cost(probs, labels_matrix):
  return scalar_multiply(-1.0/size(labels_matrix, 2), 
                          matrix_dot_product(arrayfun(np.log, probs),
                                             labels_matrix))

def gradient_calculate(delta, activations):
  W_grad = scalar_multiply(
    1.0/size(delta, 2),
    matrix_matrix4D_transpose_multiply(delta, activations))
  b_grad = scalar_multiply(1.0/size(delta, 2),
                           sum_by_row(delta)).reshape(delta.shape[0], 1)
  return (W_grad, b_grad)

################################################################################
#
# Following routines are currently NOT USED by CNN implementaion
#
# These are for generic neural net back propagation
#
################################################################################

def delta_propagate(delta_next_layer, weight_matrix, 
                    grad_fun = 0, activations = 0):
  if (grad_fun == 0):
    return filter_transpose_multiply(weight_matrix, delta_next_layer)
  else:
    return multiply(filter_transpose_multiply(weight_matrix,
                                              delta_next_layer),
                    grad_fun(activations))



def vector_sum(v):
    return reducefun(np.add, v, 1)

def matrix_sum(mat):
    return reducefun(np.add, reducefun(np.add, mat, 1), 1)

def squared_error(target, out_acts):
  #Jsqerr = (0.5/m) * ((a1 - a3) * (a1 - a3)).sum()
  err = minus(target, out_acts)
  err = sum_by_column(multiply(err, err))
  return (0.5 / size(target, 2)) * vector_sum(err)


