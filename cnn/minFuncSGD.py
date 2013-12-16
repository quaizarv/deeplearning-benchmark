import numpy
from wrappers import *

def min_func_SGD(funObj, theta_tuple, data, labels, options):
  # Runs stochastic gradient descent with momentum to optimize the
  # parameters for the given objective.
  #
  # Parameters:
  #  funObj     -  function handle which accepts as input theta,
  #                data, labels and returns cost and gradient w.r.t
  #                to theta.
  #  theta_tuple-  tuple containing parameter arrays
  #  data       -  stores data in m x n x numExamples tensor
  #  labels     -  corresponding labels in numExamples x 1 vector
  #  options    -  struct to store specific options for optimization
  #
  # Returns:
  #  theta_tuple -  optimized parameter arrays
  #
  # Options (* required)
  #  epochs*     - number of epochs through data
  #  alpha*      - initial learning rate
  #  minibatch*  - size of minibatch
  #  momentum    - momentum constant, defualts to 0.9


  #======================================================================
  # Setup
  if options.momentum < 0.9:
    options.momentum = 0.9

  epochs = options.epochs
  alpha = options.alpha
  minibatch = options.minibatch
  m = size(labels, 1)
  print m

  # Setup for momentum
  mom = 0.5
  momIncrease = 20

  (Wc, Wd, bc, bd) = theta_tuple
  Wc_velocity = zeros(size(Wc))
  Wd_velocity = zeros(size(Wd))
  bc_velocity = zeros(size(bc))
  bd_velocity = zeros(size(bd))
  velocity_tuple = (Wc_velocity, Wd_velocity, bc_velocity, bd_velocity)

  mb_data = zeros((size(data, 1), size(data, 2), minibatch))
  mb_labels = zeros((minibatch,))

  #======================================================================
  # SGD loop
  #======================================================================
  it = 0
  for e in range(epochs):
    
    # randomly permute indices of data for quick minibatch sampling
    rp = np.random.permutation(range(m))
    
    for s in range(0,m-minibatch+1, minibatch):
      it = it + 1
                        
      # increase momentum after momIncrease iterations
      if it == momIncrease:
        mom = options.momentum

      # get next randomly selected minibatch
      extract_random_minibatch(mb_data, mb_labels, data, labels,
                               rp[s:s+minibatch])
                                                                                                        
      # evaluate the objective function on the next minibatch
      (cost, _) = funObj(theta_tuple, mb_data, mb_labels, velocity_tuple,
                         alpha, mom)

      print 'Epoch %r: Cost on iteration %r is %r\n' % (e,it,cost)

    # aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0
    
  return theta_tuple

