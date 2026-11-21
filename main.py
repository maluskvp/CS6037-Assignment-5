# Assignment 5 - Implementation of the SMO Algorithm for Hard-Margin SVM


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

seed = 42

# Kernel 1 = K1: the dot product of X dot Z
def kernel1(X, Z):
  return np.dot(X,Z.T) # need the transpose so it is (n, m) x (m, n)

# Kernel 2 = K2: (K1 + 1) squared
def kernel2(X,Z):
  return (kernel1(X,Z)+1)**2

def hard_margin(kernel, alpha, y):
  return np.sum(alpha) - (0.5 * ((alpha * y) @ kernel @ (alpha * y)))



def SMO(X, y, kernel, eps=1e-8, tau=1e-3, T=10e4, seed=42):
  # SMO Part 1 initialize alpha, gram matrix Kij and
  
  #Gram matrix initialization
  K = kernel(X,X)

  m = X.shape[0]
  rng = np.random.default_rng(seed)
  ui = rng.uniform(0,1,size=m)

  a = np.zeros(m)

  a[y==1] = (ui[y==1] / np.sum(ui[y==1])) # set ai for positive
  a[y==-1] = (ui[y==-1] / np.sum(ui[y==-1])) # set ai for negative

  print("Is sum of y time a 0?", np.isclose(np.sum(a*y), 0.0, atol=eps)) # testing to check if a was computed right using epsilon as threshold

  b = 0

  iteration_history = []



  for i in range(int(T)):
    fxi = K @ (a * y) + b

    Ei = fxi - y # get the E(i) from Part 3

    yifxi = y * fxi
    yifxi = np.asarray(yifxi)
    
    # KKT(i) TODO

    # KKT violators
    vi = np.zeros_like(a)

    vi[a>0] = yifxi - 1
    vi[a<0] = np.maximum(0, 1 - yifxi[a<0])

    i1 = np.argmax(vi) # 4a get the worst KKT violator index

    if vi[i1] < tau: # use the index to check if Step 3 succeeds
      print(f"Reached KKT tolerance with vi[i1] = {vi[i1]}")
      break

    ei = Ei[i1] - Ei # get e(i) part 4b

    k = K[i1,i1] + np.diag(K) - 2 * K[i1:] #K[i1,i1] + diagnols of K to get K_ii - 2 * K[i1,i]

    valid_k_values = []
    for index, k_i in enumerate(k):
      if k_i > 0:
        valid_k_values.append(k_i) # get all k values > 0 for 4c

    if len(valid_k_values) == 0:
      print("No valid i for which k = K[i1,i1] + diagnols of K to get K_ii - 2 * K[i1,i]")
      break

    sorted_k = valid_k_values[np.argsort(np.abs(Ei[valid_k_values]))[::-1]] # sort numpy array of k indexes based on highest to lowest Ei[k] value

    for index, i2 in enumerate(sorted_k):
      e = Ei[i2] # get e for current i2 value
      k = K[i1,i1] + K[i2,i2] - 2 * K[i1,i2] # update k to be 4d
      a_i2_unc = a[index] + (y[i2] * e / k)

      a_i1_old = a[i1]
      a_i2_old = a[i2]
      l = np.max(0, a_i2_old - a_i1_old)
      h = np.inf
      if i1 == i2:
        l = 0
        h = a_i2_old + a_i1_old
      a_i2_new = np.min(np.max(a_i2_unc,l), h)


  return



def standardize_features(Xtr, Xte):
  mean = np.mean(Xtr, axis=0)
  std = np.std(Xtr, axis=0)

  std[std==0] = 1

  standardized_Xtr = (Xtr - mean) / std
  standardized_Xte = (Xte - mean) / std

  return standardized_Xtr, standardized_Xte



if __name__ == "__main__":
  # get the datasets
  # Iris_Setosa_Versicolor train and test
  df = pd.read_csv("SMO_data/iris_setosa_versicolor_train.csv")
  Xtr_ISV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_ISV = df["y"].to_numpy(dtype=float)
  df = pd.read_csv("SMO_data/iris_setosa_versicolor_test.csv")
  Xte_ISV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_ISV = df["y"].to_numpy(dtype=float)

  Xtr_ISV_stan, Xte_ISV_stan = standardize_features(Xtr_ISV, Xte_ISV)

  # Rings train and test
  df = pd.read_csv("SMO_data/rings_train.csv")
  Xtr_R = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_R = df["y"].to_numpy(dtype=float)
  df = pd.read_csv("SMO_data/rings_test.csv")
  Xte_R = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_R = df["y"].to_numpy(dtype=float)

  Xtr_R_stan, Xte_R_stan = standardize_features(Xtr_R, Xte_R)

  # XOR train and test
  df = pd.read_csv("SMO_data/iris_setosa_versicolor_train.csv")
  Xtr_XOR = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_XOR = df["y"].to_numpy(dtype=float)
  df = pd.read_csv("SMO_data/iris_setosa_versicolor_test.csv")
  Xte_XOR = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_XOR = df["y"].to_numpy(dtype=float)

  Xtr_XOR_stan, Xte_XOR_stan = standardize_features(Xtr_XOR, Xte_XOR)

  # Iris_Versicolor_Virginica train

  df = pd.read_csv("SMO_data/iris_setosa_versicolor_train.csv")
  X_tr_IVV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Y_tr_XOR = df["y"].to_numpy(dtype=float)


  # iris_setosa_versicolor train
  SMO(Xtr_ISV_stan,Ytr_ISV,kernel1,T=2)

  