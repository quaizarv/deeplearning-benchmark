import numpy
from wrappers import *
from utils import *

def cnn_cost(theta_tuple, images, labels, cnn_config,
             velocity_tuple = None, alpha = 1, mom = 1):
  """ Calculate cost and gradient for a single layer CNN

  The output layer is a softmax layer with cross entropy objective.
  
  Parameters:
  theta       -  unrolled parameter vector
  images      -  stores images in num_images x image_dim x image_dim
  array
  labels      -  image labels                     
  pred        -  boolean only forward propagate and return predictions
  grad_stack  -  stores the theta gradients in a long flat array
  cnn_config  -  configuration for a single CNN layer, e.g. various dimensions
  
  Returns:
  cost       -  cross entropy cost
  preds      -  list of predictions for each example (if pred==True)
  gradients  -  gradient with respect to theta (if pred==False) are returned
  in the grad_stack argument
  """
  num_images = cnn_config.num_images
  image_dim = cnn_config.image_dim
  num_filters = cnn_config.num_filters
  filter_dim = cnn_config.filter_dim
  pool_dim = cnn_config.pool_dim
  num_classes = cnn_config.num_classes

  # Reshape parameters and setup gradient matrices
  #
  # Wc is filter_dim x filter_dim x num_filters parameter matrix
  # bc is the corresponding bias
  #
  # Wd is num_classes x hidden_size parameter matrix where hiddenSize
  # is the number of output units from the convolutional layer
  # bd is corresponding bias
  (Wc, Wd, bc, bd) = theta_tuple

  #======================================================================
  # STEP 1a: Forward Propagation
  #
  conv_dim = image_dim-filter_dim+1  # dimension of convolved output
  out_dim = conv_dim/pool_dim     # dimension of subsampled output

  # activations: num_images x num_filters x conv_dim x conv_dim
  # pooled_activations: num_images x num_filters x ouput_dim x ouput_dim
  activations = filter_convolve(images, Wc, bc)
  pooled_activations = average_pool_T4(activations, pool_dim)

  # Softmax Layer
  #  Forward propagate the pooled activations calculated above into a
  #  standard softmax layer. 
  #
  # probs: num_classes x num_images
  probs = matrix_matrix4D_multiply(Wd, pooled_activations)
  probs = softmax(plus(probs, bd))

  #======================================================================
  # STEP 1b: Calculate Cost

  cost = 0  # save objective into cost

  # TBD: labels -> labels matrix can be done in cnnTrain.py
  labels_matrix = bit_numbers_to_bit_vectors(num_classes, labels)

  cost = softmax_cost(probs, labels_matrix)
  
  # Makes predictions given probs and returns without backproagating errors.
  preds = 0
  if (velocity_tuple == None):
    #TBD: should pred be returned as a column vector (i.e. of size (n, 1))
    # or a simply array
    preds = argmax_by_column(probs)
    return (cost, preds)
    
  #======================================================================
  # Backpropagation & Gradient Computation
    
  # Compute the error at the output layer (softmax)
  delta_softmax = minus(probs, labels_matrix)

  # Back propogate errors from softmax layer to pooling layer
  delta_pool = matrix_matrix_transpose_multiply_to_4D(
    Wd, delta_softmax, (out_dim, out_dim, num_filters, num_images))

  (Wc_velocity, Wd_velocity, bc_velocity, bd_velocity) = velocity_tuple

  # Compute Wd Gradient
  (Wd_grad, bd_grad) = gradient_calculate(delta_softmax, pooled_activations)

  # Add in the weighted velocity vector to the gradient evaluated above scaled
  # by the learning rate.  Then update the current weights theta according to
  # the sgd update rule
  Wd_velocity[:] = plus(scalar_multiply(mom,  Wd_velocity),
                        scalar_multiply(alpha, Wd_grad))
  bd_velocity[:] = plus(scalar_multiply(mom,  bd_velocity),
                        scalar_multiply(alpha, bd_grad))
  Wd[:] = minus(Wd, Wd_velocity)
  bd[:] = minus(bd, bd_velocity)

  # Back propogate errors from pooling to convolution layer
  #
  # first upsample the errors at the pooling layer and then multiply
  # by the gradient of the activations at the convolution layer
  delta_conv = multiply(upsample_T4(delta_pool, pool_dim),
                        sigmoid_gradient(activations))

  """
  delta_conv = zeros((conv_dim, conv_dim, num_filters, num_images))
  for image_num in range(0, num_images):
    for filter_num in range(0, num_filters):
      delta_p = get_matrix(delta_pool, (filter_num, image_num))
      delta_c = upsample(delta_p, pool_dim)
      acts = get_matrix(activations, (filter_num, image_num))
      set_matrix(delta_conv, (filter_num, image_num),
                 multiply(delta_c, sigmoid_gradient(acts)))
  """

  # Compute Wc Gradient
  #Wc_grad = zeros(size(Wc))
  #bc_grad = zeros(size(bc))
  #for image_num in range(0, num_images):
    #for filter_num in range(0, num_filters):
      #filter_grad = get_matrix(Wc_grad, filter_num)
      #delta_c = get_matrix(delta_conv, (filter_num, image_num))
      #image = get_matrix(images, image_num)
      #set_matrix(Wc_grad, filter_num,
      #           plus(filter_grad, convolve_2d(image, delta_c)))
      #bc_grad[filter_num][0] += delta_c.sum()
  Wc_grad = grad_conv_T4(images, delta_conv)
  bc_grad = delta_conv.sum((0, 2, 3)).reshape((num_filters, 1))

  Wc_grad = scalar_multiply(1.0/num_images, Wc_grad)
  bc_grad = scalar_multiply(1.0/num_images, bc_grad)
    
  Wc_velocity[:] = plus(scalar_multiply(mom,  Wc_velocity),
                        scalar_multiply(alpha, Wc_grad))
  bc_velocity[:] = plus(scalar_multiply(mom,  bc_velocity),
                        scalar_multiply(alpha, bc_grad))
  Wc[:] = minus(Wc, Wc_velocity)
  bc[:] = minus(bc, bc_velocity)
  
  return (cost, preds)
  

def cnn_cost_with_gradient(theta, images, labels, cnn_config, preds = False):

  theta_tuple = array_to_stack(theta, cnn_config)

  velocity_array = None
  velocity_tuple = None
  if (preds == False):
    velocity_array = zeros(size(theta))
    velocity_tuple = array_to_stack(velocity_array, cnn_config)

  (cost, _) = cnn_cost(theta_tuple, images, labels, cnn_config, 
                       velocity_tuple, 1, 1)
  
  return (cost, velocity_array)
