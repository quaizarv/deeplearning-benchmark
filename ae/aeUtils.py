from utils import *
from wrappers import *

def ae_array_to_stack(param_stack, ae_config):
  """ Unroll flat array into paramter matrices/vectors
  """
  visible_size = ae_config.visible_size
  hidden_size = ae_config.hidden_size

  start = 0
  (W1, start) = get_array_view(param_stack, start, 
                               (hidden_size, visible_size))
  (W2, start) = get_array_view(param_stack, start, (visible_size, hidden_size))
  (b1, start) = get_array_view(param_stack, start, (hidden_size, 1))
  (b2, start) = get_array_view(param_stack, start, (visible_size, 1))
  return (W1, W2, b1, b2)

def sparse_cost(beta, rho, rho_hat):
  """ Compute Sparsity Penalty based on KL-divergence
  """
  return beta * matrix_sum(
    arrayfun(
      lambda elem: rho*np.log(rho/elem) + (1-rho)*np.log((1-rho)/(1-elem)),
      rho_hat))

