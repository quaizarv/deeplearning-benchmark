from utils import *
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

