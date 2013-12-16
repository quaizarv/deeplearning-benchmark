import struct
from wrappers import *

def load_MNIST_labels(filename):

  """Read MNIST image labels from a file into a matrix
  
  Parameters
    filename: Filename to read labels from
     
  Returns
    images:  [number of MNIST images]x1 matrix containing the MNIST labels
  """
  try:
    fp = open(filename, 'rb')
    (magic,) = struct.unpack('>i', fp.read(4))
    assert magic == 2049, 'Bad magic number in %r' % filename
    (num_labels,) = struct.unpack('>i', fp.read(4))

    labels_str = fp.read()
    assert len(labels_str) == num_labels, "Mismatch in label count"
    labels = np.array(struct.unpack(str(len(labels_str)) + 'B',
                                    labels_str[0:len(labels_str)]))

    #convert to appropriate float size 
    return labels.astype(floatX)

  finally:
    fp.close()
