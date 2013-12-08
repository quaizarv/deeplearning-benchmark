import scipy.io
import numpy as np

## ---------------------------------------------------------------

def normalize_data(patches):

  """Squash data to [0.1, 0.9] since we use sigmoid as the activation
  function in the output layer
  """

  # Remove DC (mean of images). 
  patches = patches - patches.mean(0)

  # Truncate to +/-3 standard deviations and scale to -1 to 1
  pstd = 3 * patches.std()
  patches = np.maximum(np.minimum(patches, pstd), -pstd) / pstd

  # Rescale from [-1,1] to [0.1,0.9]
  patches = (patches + 1) * 0.4 + 0.1

  return patches

def sample_images():
  """Returns 10000 patches for training as a [8*8, 10000] matrix
  """

  images = scipy.io.loadmat("IMAGES.mat")['IMAGES'] # load images from disk 

  #  images is a 3D array containing images, e.g. (512, 512, 10) contains
  #  10 images of size 512 x 512
  #
  #  For instance, images(:,:,6) is a 512x512 array containing the 6th image,
  #  and you can type "imagesc(images(:,:,6)), colormap gray;" to visualize
  #  it. (The contrast on these images look a bit off because they have
  #  been preprocessed using using "whitening."  

  patch_size = 8        # we'll use 8x8 patches 
  num_patches = 10000

  # take a random sample of 10000 patches of 8x8 - return a matrix 
  patches = np.zeros((patch_size*patch_size, num_patches))
  for i in range(num_patches):
    image_num = np.random.random_integers(0, 9)
    image_pixels = images[:,:,image_num]
    patch_start = np.random.random_integers(0, images.shape[0] - patch_size)
    patch_stop = patch_start + patch_size
    patch = image_pixels[patch_start:patch_stop,patch_start:patch_stop]
    patch = patch.reshape((patch_size ** 2, ))
    patches[:, i] = patch


  ## ---------------------------------------------------------------
  # For the autoencoder to work well we need to normalize the data
  # Specifically, since the output of the network is bounded between [0,1]
  # (due to the sigmoid activation function), we have to make sure 
  # the range of pixel values is also bounded between [0,1]
  patches = normalize_data(patches)
  return patches
