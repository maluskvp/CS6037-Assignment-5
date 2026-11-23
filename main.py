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

    #vi[a>0] = yifxi - 1
    #vi[a<0] = np.maximum(0, 1 - yifxi[a<0])
    vi[a > 0] = np.abs(yifxi[a > 0] - 1)
    vi[a == 0] = np.maximum(0, 1 - yifxi[a == 0])
    
    i1 = np.argmax(vi) # 4a get the worst KKT violator index

    if vi[i1] < tau: # use the index to check if Step 3 succeeds
      print(f"Reached KKT tolerance with vi[i1] = {vi[i1]}")
      break

    ei = Ei[i1] - Ei # get e(i) part 4b

    k = K[i1,i1] + np.diag(K) - 2 * K[i1, :] #K[i1,i1] + diagnols of K to get K_ii - 2 * K[i1,i]
    
    #valid_k_values = []
    valid_k_values = np.where(k > 1e-12)[0]
    valid_k_values = valid_k_values[valid_k_values != i1]

    if len(valid_k_values) == 0:
          print("No valid i for which k = K[i1,i1] + diagnols of K to get K_ii - 2 * K[i1,i]")
          break

    sorted_k = valid_k_values[np.argsort(np.abs(Ei[i1] - Ei[valid_k_values]))[::-1]] # sort numpy array of k indexes based on highest to lowest Ei[k] value

    for index, i2 in enumerate(sorted_k):
      #e = Ei[i2] # get e for current i2 value
      e = Ei[i1] - Ei[i2]
      k = K[i1, i1] + K[i2, i2] - 2 * K[i1, i2] # update k to be 4d
      a_i1_old = a[i1]
      a_i2_old = a[i2]
      a_i2_unc = a_i2_old + y[i2] * e / k
      if y[i1] != y[i2]:
          l = max(0, a_i2_old - a_i1_old)
          h = np.inf
      else:
          l = 0
          h = a_i2_old + a_i1_old

      a_i2_new = np.clip(a_i2_unc, l, h)
      # then to update alpha1 while keeping the equality constraint
      a_i1_new = a_i1_old + y[i1] * y[i2] * (a_i2_old - a_i2_new)
    
      if abs(a_i2_new - a_i2_old) < eps: # skip updates that are too small
          continue

      if a_i1_new < -eps:
          continue
      b_old = b # update the bias using the two changed points

      b1 = (
          b_old - Ei[i1]
          - y[i1] * (a_i1_new - a_i1_old) * K[i1, i1]
          - y[i2] * (a_i2_new - a_i2_old) * K[i1, i2]
      )

      b2 = (
          b_old - Ei[i2]
          - y[i1] * (a_i1_new - a_i1_old) * K[i1, i2]
          - y[i2] * (a_i2_new - a_i2_old) * K[i2, i2]
      )

      # to save the updated alpha values
      a[i1] = a_i1_new
      a[i2] = a_i2_new

      # setting tiny alpha values to zero
      a[a < eps] = 0

      # choose the bias from the updated support vectors
      if a[i1] > eps:
          b = b1
      elif a[i2] > eps:
          b = b2
      else:
          b = (b1 + b2) / 2

      break

    # record the objective and training progress
    fxi = K @ (a * y) + b
    predictions = np.where(fxi >= 0, 1, -1)

    objective = hard_margin(K, a, y)
    train_errors = np.sum(predictions != y)

    iteration_history.append({
        "iteration": i + 1,
        "objective": objective,
        "train_errors": int(train_errors),
        "support_vectors": int(np.sum(a > eps)),
        "equality": float(np.sum(a * y)),
        "min_alpha": float(np.min(a)),
        "max_alpha": float(np.max(a))
    })

  #return the trained model and history
  print("iterations:", len(iteration_history))
  print("support vectors:", np.sum(a > eps))
  print("sum(y * alpha):", np.sum(a * y))
  print("min(alpha):", np.min(a))

  return a, b, iteration_history



def standardize_features(Xtr, Xte):
  mean = np.mean(Xtr, axis=0)
  std = np.std(Xtr, axis=0)

  std[std==0] = 1

  standardized_Xtr = (Xtr - mean) / std
  standardized_Xte = (Xte - mean) / std

  return standardized_Xtr, standardized_Xte



if __name__ == "__main__":
  # get the datasets
  # fixing the dataset used:
  # Iris_Setosa_Versicolor train and test
  df = pd.read_csv("SMO_data/iris_setosa_versicolor_train.csv")
  Xtr_ISV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_ISV = df["y"].to_numpy(dtype=float)

  df = pd.read_csv("SMO_data/iris_setosa_versicolor_test.csv")
  Xte_ISV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_ISV = df["y"].to_numpy(dtype=float)

  Xtr_ISV_stan, Xte_ISV_stan = standardize_features(Xtr_ISV, Xte_ISV)
  # XOR train and test
  df = pd.read_csv("SMO_data/xor_train.csv")
  Xtr_XOR = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_XOR = df["y"].to_numpy(dtype=float)

  df = pd.read_csv("SMO_data/xor_test.csv")
  Xte_XOR = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_XOR = df["y"].to_numpy(dtype=float)

  Xtr_XOR_stan, Xte_XOR_stan = standardize_features(Xtr_XOR, Xte_XOR)

  # iris versicolor vs virginica
  df = pd.read_csv("SMO_data/iris_versicolor_virginica.csv")
  X_tr_IVV = df.drop(columns=["y"]).to_numpy(dtype=float)
  Y_tr_IVV = df["y"].to_numpy(dtype=float)

  # standardize the dataset
  IVV_mean = np.mean(X_tr_IVV, axis=0)
  IVV_std = np.std(X_tr_IVV, axis=0)
  IVV_std[IVV_std == 0] = 1

  X_tr_IVV_stan = (X_tr_IVV - IVV_mean) / IVV_std

  a, b, history = SMO(
    Xtr_ISV_stan,
    Ytr_ISV,
    kernel1,
    T=10000,
    seed=42
)

print("bias:", b)
print("final objective:", history[-1]["objective"] if history else "no updates")

# check part a results
print("all alpha >= 0:", np.all(a >= -1e-8))

predictions = np.sign(
    kernel1(Xtr_ISV_stan, Xtr_ISV_stan) @ (a * Ytr_ISV) + b
)

print("training accuracy:", np.mean(predictions == Ytr_ISV))

objectives = [h["objective"] for h in history]
print("objective non-decreasing:",
      np.all(np.diff(objectives) >= -1e-6))

# test part a with polynomial kernel
a2, b2, history2 = SMO(
    Xtr_ISV_stan,
    Ytr_ISV,
    kernel2,
    T=10000,
    seed=42
)

print("\n--- kernel 2 results ---")
print("bias:", b2)
print("iterations:", len(history2))
print("support vectors:", np.sum(a2 > 1e-8))
print("alpha nonnegative:", np.all(a2 >= -1e-8))

predictions2 = np.sign(
    kernel2(Xtr_ISV_stan, Xtr_ISV_stan) @ (a2 * Ytr_ISV) + b2
)

print("training accuracy:", np.mean(predictions2 == Ytr_ISV))

# part c1 which is compare 10 random initializations
print("\n--- part c1: random initializations ---")

for seed in range(10):
    alpha, bias, history = SMO(
        Xtr_ISV_stan,
        Ytr_ISV,
        kernel1,
        T=10000,
        seed=seed
    )

    w = (alpha * Ytr_ISV) @ Xtr_ISV_stan

    print(
        f"seed {seed}: "
        f"w={np.round(w, 4)}, "
        f"b={bias:.4f}, "
        f"sv={np.sum(alpha > 1e-8)}"
    )


# part c2 to compare our smo with sklearn svc
print("\n--- part c2: sklearn comparison ---")

for name, kernel_func, svc_model in [
    ("K1", kernel1, SVC(kernel="linear", C=1e8)),
    ("K2", kernel2, SVC(kernel="poly", degree=2, gamma=1, coef0=1, C=1e8))
]:
    alpha, bias, history = SMO(
        Xtr_ISV_stan, Ytr_ISV, kernel_func, T=10000, seed=42
    )

    svc_model.fit(Xtr_ISV_stan, Ytr_ISV)

    # sklearn stores signed dual coefficients
    sklearn_alpha = np.zeros(len(Ytr_ISV))
    sklearn_alpha[svc_model.support_] = (
        np.abs(svc_model.dual_coef_[0])
    )

    K = kernel_func(Xtr_ISV_stan, Xtr_ISV_stan)

    print(f"\n{name}:")
    print("our bias is :", bias)
    print("sklearn bias is :", svc_model.intercept_[0])
    print("our support vectors are:", np.where(alpha > 1e-8)[0])
    print("sklearn support vectors are:", svc_model.support_)
    print("our objective is:", hard_margin(K, alpha, Ytr_ISV))
    print("sklearn objective is:", hard_margin(K, sklearn_alpha, Ytr_ISV))

# part c3 this to verify constraints during training
print("\n--- part c3: constraint verification ---")

for name, kernel_func in [("K1", kernel1), ("K2", kernel2)]:
    alpha, bias, history = SMO(
        Xtr_ISV_stan, Ytr_ISV, kernel_func, T=10000, seed=42
    )

    min_alpha = min(h["min_alpha"] for h in history)
    max_equality = max(abs(h["equality"]) for h in history)

    print(f"{name}:")
    print("minimum alpha:", min_alpha)
    print("maximum equality error:", max_equality)
    print("alpha nonnegative:", min_alpha >= -1e-8)
    print("equality satisfied:", max_equality < 1e-8)


# part d to test nonseparable datasets
print("\n--- part d: hard-margin failure ---")

cases = [
    ("XOR K1", Xtr_XOR_stan, Ytr_XOR, kernel1),
    ("Iris IVV K1", X_tr_IVV_stan, Y_tr_IVV, kernel1),
    ("Iris IVV K2", X_tr_IVV_stan, Y_tr_IVV, kernel2)
]

for name, X, y, kernel_func in cases:
    print(f"\n{name}")

    alpha, bias, history = SMO(
        X, y, kernel_func, T=1000, seed=42
    )

    if not history:
        print("no iterations recorded")
        continue

    iterations = [h["iteration"] for h in history]
    max_alpha = [h["max_alpha"] for h in history]
    objectives = [h["objective"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(iterations, max_alpha)
    axes[0].set_title(f"{name}: max alpha")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("max alpha")

    axes[1].plot(iterations, objectives)
    axes[1].set_title(f"{name}: dual objective")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("W(alpha)")

    plt.tight_layout()
    plt.savefig(f"part_d_{name.replace(' ', '_')}.png")
    plt.show()
