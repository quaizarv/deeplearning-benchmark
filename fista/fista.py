import time
import scipy
import numpy as np
from samplePatches import *
#from buildDict import *
from numpy.linalg import norm

def shrink(x, T):
  """ Compute Shrinkage function, sign(x)(|abs(x)| - threshold)
  """
  if (np.sum(abs(T)) == 0):
    y = x
  else:
    y = np.maximum(abs(x) - T, 0)
    y = np.sign(x) * y
  return y

def count_non_sparse_codes(S, num_features):
  return ((abs(S) > 1e-10).sum(1) > (num_features/10.0)).sum()

def count_valid_sparse_codes(S, num_features):
  return ((abs(S) < 1e-10).sum(1) > (num_features/20.0)).sum()

def non_zero_elems(s, m):
  """ Count non-zero elements in a sparse code vector
  """
  return (abs(s) > 0).sum()/m

def fista_solve(W, inp_vec):
  """Convert input vector to sparse code using FISTA

     Parameters:

     W - (input_dim x num_features) dictionary (feature vectors)
     inp_vec - (input_dim x num_samples) input vector
  """
  maxIter = 100
  x0 = np.zeros((W.shape[1],1))
  num_features = W.shape[1]
  m = inp_vec.shape[1]

  ## Initializing optimization variables
  t_k = 1  
  t_km1 = 1 

  # Compute W'W where W is the dictionary
  G = np.dot(np.transpose(W), W)
  
  # compute L as the largest eigen value of W'W - L will be used as 
  # in computing the threshold for the shrinkage function
  (U, D, V) = np.linalg.svd(G);
  L0 = D[0]
  #L0 = 1

  # compute W'x
  c = np.dot(np.transpose(W), inp_vec)
  # sk = zeros(n,1) 
  (sk, _, _, _) = np.linalg.lstsq(W, inp_vec)

  #L0 = 1 
  #lamda = 0.5*norm(c,np.inf) 
  lamda = 1
  # lamda_bar = 1e-10*lamda0 
  lamda_bar = 1e-2
  L = L0 

  print "L ", L

  eta = 0.99

  skm1 = sk
  nIter = 0 
  while (nIter < maxIter):
    nIter = nIter + 1 

    # update the sparse code using momentum
    yk = sk + ((t_km1-1)/t_k)*(sk-skm1) 
    
    # Compute gradient of the reconstruction erro
    grad = np.dot(G, yk) - c  
    
    # Compute new sparse code by applying the shrinkage function
    skp1 = yk - (1/L)*grad
    skp1 = shrink(skp1, lamda/L) 

    # vary the coefficient for the sparsity penalty depending on
    # how sparse the resulting code is
    nz_count = non_zero_elems(skp1, m)
    if (nz_count > num_features/10.0):
      lamda = lamda/0.5
    elif (nz_count < num_features/5.0):
      lamda = lamda*0.5
    else:
      lamda = lamda
      #lamda = 2 * lamda 
      #else:
    #lamda = max(eta*lamda,lamda_bar) 

    # Compute new momentum parameters
    t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k)) 
    t_km1 = t_k 
    t_k = t_kp1 
    skm1 = sk 
    sk = skp1 


    #cost1 = 0.5*(norm(inp_vec - np.dot(W, sk))**2)
    #cost2 = norm(sk,1)

  nz_count = non_zero_elems(sk, m)
  cost1 = (0.5/m)*((inp_vec - np.dot(W, sk))**2).sum()
  cost2 = (1.0/m)*abs(sk).sum()
  print  "cost: ", cost1, cost2,
  print "lamda and # of non-zero elems: ", lamda, nz_count, "\n"
  return sk


def run_fista_sc(X, num_features, iters, lamda):
  """Learn a Dictionary of features from input data X using FISTA for Sparse
     Coding.

     Dictionary is initialized randomly. We learn the dictionary by looping
     over the following 2 steps:

     1. Fix dictionary and map X to sparse codes using FISTAN

     2. Compute dictionary by solving Dict x S = X (or rather S' x Dict' = X')

     Parameters:
      X:     Input data as (num_patches, input_dim) matrix
      D:     Distonary of features

  """

  # initialize dictionary
  dict = np.random.randn(X.shape[0], num_features)
  dict = dict / np.sqrt((dict**2).sum(0) + 1e-20).reshape(1, dict.shape[1])
  t1 = time.time()
  for itr in range(iters):
    print "Running sparse coding: iteration =",  itr
    S = fista_solve(dict, X)
    (dict1, _, _, _) = np.linalg.lstsq(np.transpose(S), np.transpose(X))
    dict = np.transpose(dict1)
    dict = dict / np.sqrt((dict**2).sum(0) + 1e-20).reshape(1, dict.shape[1])
  print time.time() - t1
  return dict

def test_fista():
  images = scipy.io.loadmat("IMAGES.mat")['IMAGES'] # load images from disk 
  patches = sample_images(images, 8, 20000)
  W = run_fista_sc(patches, 121, 40, 0.015)