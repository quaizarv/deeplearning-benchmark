from wrappers import *

def compute_numerical_gradient(J, theta):
  num_grad = zeros(size(theta, 1))
  epsilon = 1e-4

  for i in range(size(theta, 1)):
    oldT = theta[i]
    theta[i] = oldT+epsilon
    (pos, _) = J(theta)
    theta[i] = oldT-epsilon
    (neg, _) = J(theta)
    num_grad[i] = (pos-neg)/(2*epsilon)
    theta[i] = oldT
    if ((i+1) % 100) == 0:
      print 'Done with %r\n' % i
      
  return num_grad
