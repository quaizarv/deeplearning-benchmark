import time
import scipy
import numpy as np
from samplePatches import *
from buildDict import *
from numpy.linalg import norm

def shrink(x, T):
  if (np.sum(abs(T)) == 0):
    y = x
  else:
    y = np.maximum(abs(x) - T, 0)
    y = np.sign(x) * y
  return y

def solve_fista(W, inp_vec):
  """Convert input vector to sparse code using FISTA

  inp_vec - (input_dim x 1) input vector
  W - (input_dim x num_features) dictionary (feature vectors)
  """
  maxIter = 500
  x0 = np.zeros((W.shape[1],1))

  ## Initializing optimization variables
  t_k = 1  
  t_km1 = 1 

  G = np.dot(np.transpose(W), W)
  (U, D, V) = np.linalg.svd(G);
  L0 = D[0]
  #L0 = 1
  c = np.dot(np.transpose(W), inp_vec)
  # sk = zeros(n,1) 
  (sk, _, _, _) = np.linalg.lstsq(W, inp_vec)

  #L0 = 1 
  lamda = 0.5*norm(c,np.inf) 
  # lamda_bar = 1e-10*lamda0 
  lamda_bar = 1e-6
  L = L0 

  print "L ", L

  eta = 0.95 

  skm1 = sk
  nIter = 0 
  while (nIter < maxIter):
    nIter = nIter + 1 
    yk = sk + ((t_km1-1)/t_k)*(sk-skm1) 
    grad = np.dot(G, yk) - c  # gradient of cost function at yk
    
    skp1 = yk - (1/L)*grad
    skp1 = shrink(skp1, lamda/L) 

    lamda = max(eta*lamda,lamda_bar) 
    t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k)) 
    t_km1 = t_k 
    t_k = t_kp1 
    skm1 = sk 
    sk = skp1 

    cost1 = 0.5*(norm(inp_vec - np.dot(W, sk))**2)
    cost2 = norm(sk,1)
    if (nIter % 20 == 0):
      print "iteration: ", nIter, " cost: ", cost1, cost2
  return sk


def test_fista():
  images = scipy.io.loadmat("IMAGES.mat")['IMAGES'] # load images from disk 
  patches = sample_images(images, 8, 20)
  print patches.shape
  W = load_dict()
  t1 = time.time()
  scode = solve_fista(np.transpose(W), patches[:, 0])
  print time.time() - t1
