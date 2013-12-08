import struct
import numpy
from wrappers import *

def load_MNIST_images(filename):
  """Read MNIST images from a file into a matrix
  
  Parameters
    filename: Filename to read images from
     
  Returns
    images:  [number of MNIST images]x28x28 matrix containing the raw MNIST
             images
  """
  try:
    fp = open(filename, 'rb')

    (magic,) = struct.unpack('>i', fp.read(4))
    assert magic == 2051, 'Bad magic number in %r' % filename
    (num_images,) = struct.unpack('>i', fp.read(4))
    (num_rows,) = struct.unpack('>i', fp.read(4))
    (num_cols,) = struct.unpack('>i', fp.read(4))

    images_str = fp.read()
    images = np.array(struct.unpack(str(len(images_str)) + 'B',
                                    images_str[0:len(images_str)]))

    reshape(images, (num_rows, num_cols, num_images))
    images = np.transpose(images, (0, 2, 1))

    # Convert to float32 and rescale to [0,1]
    images = images.astype(floatX) / 255
    return images

  finally:
    fp.close()

#except IOError:
#  assert True, "File: %r not found" % filename

  
