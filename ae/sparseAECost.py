import numpy as np
from utils import *
from wrappers import *
from aeUtils import *

def sparse_ae_cost(params_array, ae_config, data, cost_only = False):
  """compute cost and gradient for an Auto Encoder

  ae_config: collection of a parameters
    visibleSize: the number of input units (probably 64) 
    hiddenSize: the number of hidden units (probably 25) 
    lamda: weight decay parameter
    sparsityParam: The desired average activation for the hidden units
    beta: weight of sparsity penalty term
    data: Our 10000x64 matrix containing the training data.
          So, data(i, :) is the i-th training example. 
  
    The input param_stack is a vector (because minFunc expects the parameters to
    be a vector).  We first convert theta to the (W1, W2, b1, b2) matrix/vector
    format, so that this follows the notation convention of the lecture notes.

  """

  visible_size = ae_config.visible_size
  hidden_size = ae_config.hidden_size
  rho = ae_config.sparsity_param
  lamda = ae_config.lamda
  beta = ae_config.beta
  (W1, W2, b1, b2) = ae_array_to_stack(params_array, ae_config)

  m = size(data,2)
  a1 = data
  z2 = filter_multiply(W1, b1, a1)
  a2 = soft_threshold(z2)
  z3 = filter_multiply(W2, b2, a2)
  a3 = soft_threshold(z3)
  rho_hat = scalar_multiply((1.0/m), sum_by_row(a2))
  reshape(rho_hat, (size(rho_hat, 1), 1))
  Jspars = sparse_cost(beta, rho, rho_hat)
  Jsqerr = squared_error(a1, a3)
  Jreg = (lamda/2.0) * (matrix_sum(multiply(W2, W2)) + matrix_sum(multiply(W1, W1)))
  cost = Jsqerr + Jreg + Jspars
  if (cost_only):
    return (cost, None)

  delta3 = (a3 - a1) * sigmoid_gradient(a3)
  W2grad = (1.0/m) * np.dot(delta3, np.transpose(a2))
  #add regularization to weights
  W2grad = plus(W2grad, scalar_multiply(lamda, W2))
  b2grad = scalar_multiply(1.0/m, sum_by_row(delta3))
  reshape(b2grad, (size(delta3, 1), 1))

  delta2 = multiply(plus(matrix_transpose_matrix_multiply(W2, delta3), 
                         scalar_multiply(beta, 
                                         arrayfun(lambda elem: 
                                                  -rho/elem + (1-rho)/(1-elem),
                                                  rho_hat))),
                    sigmoid_gradient(a2))
  W1grad = scalar_multiply(1.0/m, matrix_matrix_transpose_multiply(delta2, a1))
  W1grad = plus(W1grad, scalar_multiply(lamda, W1))
  b1grad = scalar_multiply(1.0/m, sum_by_row(delta2))
  reshape(b1grad, (size(delta2, 1), 1))

  grad_array = zeros(size(params_array))
  (W1_g, W2_g, b1_g, b2_g) = ae_array_to_stack(grad_array, ae_config)
  W1_g[:] = W1grad
  W2_g[:] = W2grad
  b1_g[:] = b1grad
  b2_g[:] = b2grad

  return (cost, grad_array)
