import numpy as np
import matplotlib.pyplot as plt

def display_network(A, cols):
  """Visualizes filters in matrix A. 

  Each row of A is a filter. We will reshape each column into a square image
  and visualizes on each cell of the visualization panel.  All other parameters
  are optional, usually you do not need to worry about it.
 
  cols:how many columns are there in the display. Default value is the
  squareroot of the number of columns in A.
  """

  # rescale
  A = A - A.mean()

  # compute rows, cols
  (M, L)= A.shape
  sz = np.sqrt(L)
  buf=1
  n = cols
  m = np.floor(M/n).astype(int)

  arr = -np.ones((buf+m*(sz+buf),buf+n*(sz+buf)))

  k=0;
  for i in range(m):
    for j in range(n):
      clim=max(abs(A[k]));
      start0 = buf + i*(sz+buf)
      start1 = buf + j*(sz+buf)
      arr[start0:start0+sz,start1:start1+sz] = A[k].reshape((sz,sz))/clim
      k=k+1

  plt.imshow(arr,cmap='gray', vmin=-1, vmax=1)
  plt.show()
  return arr
