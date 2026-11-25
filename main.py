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
      print(f"Converged - reached KKT tolerance with vi[i1] = {vi[i1]}")
      break

    ei = Ei[i1] - Ei # get e(i) part 4b

    k = K[i1,i1] + np.diag(K) - 2 * K[i1, :] #K[i1,i1] + diagnols of K to get K_ii - 2 * K[i1,i]
    
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


def getGridAndDecisionFunction(Xtr, Ytr, kernel, alpha, bias):

  # get the max and min for the x value (1st column in each csv)
  xmin = Xtr[:,0].min() - 1 # - 1 for the margin
  xmax = Xtr[:,0].max() + 1 # + 1 for the margin

  # get the max and min for the y value (2nd column in each csv)
  ymin = Xtr[:,1].min() - 1 # - 1 for the margin
  ymax = Xtr[:,1].max() + 1 # + 1 for the margin

  # create grid from margins
  xrange = np.linspace(xmin, xmax, 300) # create 300 points between xmin and xmax
  yrange = np.linspace(ymin, ymax, 300) # create 300 points between ymin and ymax

  xx, yy = np.meshgrid(xrange, yrange) # this will be used for the contour

  grid = np.c_[xx.ravel(), yy.ravel()] # flatten the 2D into x,y values so we can use this for our decision function

  # get the decision matrix over the grid
  decision_function = []
  for g in grid: 
    fx = 0
    for index, x in enumerate(Xtr):
      fx += alpha[index] * Ytr[index] * kernel(x,g) # iterate through all Xtr values and append to the fx 
    fx += bias # add the bias after taking the sum of all the product values based on index and x for decision function
    decision_function.append(fx) # add decision function values to match the grid size

  return xx, yy, np.array(decision_function)

def getConfusionMatrixAndMargin(Xtr,Ytr,Xte,Yte, kernel, alpha, bias):

  # get decision function over Xtr
  Xtr_fx = []
  for i in Xtr: 
    fx = 0
    for index, x in enumerate(Xtr):
      fx += alpha[index] * Ytr[index] * kernel(x,i) # iterate through all Xtr values and append to the fx 
    fx += bias # add the bias after taking the sum of all the product values based on index and x for decision function
    Xtr_fx.append(fx) # add decision function values to match the grid size

  Ytr_predictions = []
  for x in Xtr_fx:
    if x >= 0:
      Ytr_predictions.append(1) # predict 1 if decision function at x is >= 0 based on decision boundary in b1
    else:
      Ytr_predictions.append(-1) # predict -1 otherwise

  # Repeat for Xte
  Xte_fx = []
  for i in Xte: 
    fx = 0
    for index, x in enumerate(Xtr):
      fx += alpha[index] * Ytr[index] * kernel(x,i) # iterate through all Xte values and append to the fx 
    fx += bias # add the bias after taking the sum of all the product values based on index and x for decision function
    Xte_fx.append(fx) # add decision function values to match the grid size

  Yte_predictions = []
  for x in Xte_fx:
    if x >= 0:
      Yte_predictions.append(1) # predict 1 if decision function at x is >= 0 based on decision boundary in b1
    else:
      Yte_predictions.append(-1) # predict -1 otherwise

  Xtr_confusion = confusion_matrix(Ytr, Ytr_predictions, labels=[-1,1])
  Xte_confusion = confusion_matrix(Yte, Yte_predictions, labels=[-1,1])

  print("\nTrain Confusion Matrix")
  print(Xtr_confusion)

  print("\nTest Confusion Matrix")
  print(Xte_confusion)

  print("\nTraining Accuracy = ", accuracy_score(Ytr,Ytr_predictions))
  print("Test Accuracy = ", accuracy_score(Yte, Yte_predictions))

  # first get ||w||^2

  w2 = 0.0
  
  for i, xi in enumerate(Xtr):
     for j, xj in enumerate(Xtr):
        w2 += alpha[i] * alpha[j] * Ytr[i] * Ytr[j] * kernel(xi,xj)

  w = np.sqrt(w2)

  margin = 2/w
  sum_a = np.sum(alpha)

  print(f"Margin 2/||w|| = {margin}")
  print(f"||w||^2 = {w2}")
  print(f"sum of alpha = {sum_a}")
  print(f"w2 agrees with alpha sum? {np.isclose(w2,sum_a, atol=1e-3)}") # use isclose with the tolerance being tau to check agreement
        

def plotWaAndMisclass(history, pltTitle, supTitle):
  #plot history W(alpha) and misclassifications
  iterations = [h["iteration"] for h in history]
  Walpha = [h["objective"] for h in history]
  misclassifications = [h["train_errors"] for h in history]

  fig, (ax1, ax2) = plt.subplots(1,2)
  ax1.plot(iterations,Walpha, label="W(alpha)")
  ax1.set_xlabel("# of Iterations")
  ax1.set_ylabel("Objective W(alpha)")
  ax1.set_title("W(alpha) vs Iterations")

  ax2.plot(iterations,misclassifications, label="Misclassification")
  ax2.set_xlabel("# of Iterations")
  ax2.set_ylabel("Misclassifications")
  ax2.set_title("Misclassifications vs Iterations")
  fig.suptitle(supTitle)
  plt.savefig(pltTitle)
  plt.show()

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

  df = pd.read_csv("SMO_data/rings_train.csv")
  Xtr_R = df.drop(columns=["y"]).to_numpy(dtype=float)
  Ytr_R = df["y"].to_numpy(dtype=float)

  df = pd.read_csv("SMO_data/rings_test.csv")
  Xte_R = df.drop(columns=["y"]).to_numpy(dtype=float)
  Yte_R = df["y"].to_numpy(dtype=float)

  Xtr_R_stan, Xte_R_stan = standardize_features(Xtr_R, Xte_R)

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


# part a for toy dataset sanity test
print("\n--- part a: toy dataset test ---")

# two points that are easy to separate
X_toy = np.array([[-1.0, 0.0], [1.0, 0.0]])
y_toy = np.array([-1.0, 1.0])

alpha, bias, history = SMO(
    X_toy, y_toy, kernel1, T=1000, seed=42
)

w = (alpha * y_toy) @ X_toy
scores = X_toy @ w + bias

print("alpha:", alpha)
print("w:", w)
print("bias:", bias)
print("margin values:", y_toy * scores)

print("alpha correct:", np.allclose(alpha, [0.5, 0.5], atol=1e-3))
print("w correct:", np.allclose(w, [1.0, 0.0], atol=1e-3))
print("bias correct:", np.isclose(bias, 0.0, atol=1e-3))
print("margin constraints:", np.all(y_toy * scores >= 1 - 1e-3))
print("equality constraint:", np.isclose(np.sum(alpha * y_toy), 0.0, atol=1e-8))


# part b1 

print("\n--- part b: experiments ---")

print("Dataset = iris_setosa_versicolor_train Kernel = K1")
alpha, bias, history = SMO(Xtr_ISV_stan,Ytr_ISV,kernel1)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_ISV_stan, Ytr_ISV, kernel1, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_ISV_stan[Ytr_ISV==1, 0],Xtr_ISV_stan[Ytr_ISV==1,1], label="Setosa", marker="o") # scatter plot the Setosa values
plt.scatter(Xtr_ISV_stan[Ytr_ISV==-1, 0],Xtr_ISV_stan[Ytr_ISV==-1,1], label="Versicolor", marker="o") # scatter plot the Versicolor values
plt.scatter(Xtr_ISV_stan[alpha > eps, 0],Xtr_ISV_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("Iris Setosa Versicolor Kernel 1")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.grid(True)
plt.savefig("part_b_ISV_K1.png")
plt.show()


getConfusionMatrixAndMargin(Xtr_ISV_stan, Ytr_ISV,Xte_ISV_stan,Yte_ISV,kernel1,alpha,bias)

plotWaAndMisclass(history, "part_b_ISV_K1_Wa_Misclass.png", "Iris Setosa Versicolor Dataset Kernel 1")

print("\nDataset = iris_setosa_versicolor_train Kernel = K2")
alpha, bias, history = SMO(Xtr_ISV_stan,Ytr_ISV,kernel2)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_ISV_stan, Ytr_ISV, kernel2, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_ISV_stan[Ytr_ISV==1, 0],Xtr_ISV_stan[Ytr_ISV==1,1], label="Setosa", marker="o") # scatter plot the Setosa values
plt.scatter(Xtr_ISV_stan[Ytr_ISV==-1, 0],Xtr_ISV_stan[Ytr_ISV==-1,1], label="Versicolor", marker="o") # scatter plot the Versicolor values
plt.scatter(Xtr_ISV_stan[alpha > eps, 0],Xtr_ISV_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("Iris Setosa Versicolor Kernel 2")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.grid(True)
plt.savefig("part_b_ISV_K2.png")
plt.show()

getConfusionMatrixAndMargin(Xtr_ISV_stan, Ytr_ISV,Xte_ISV_stan,Yte_ISV,kernel2,alpha,bias)

plotWaAndMisclass(history, "part_b_ISV_K2_Wa_Misclass.png", "Iris Setosa Versicolor Dataset Kernel 2")


print("\nDataset = rings Kernel = K1")
alpha, bias, history = SMO(Xtr_R_stan,Ytr_R,kernel1)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_R_stan, Ytr_R, kernel1, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_R_stan[Ytr_R==1, 0],Xtr_R_stan[Ytr_R==1,1], label="x1", marker="o") # scatter plot the x1 values
plt.scatter(Xtr_R_stan[Ytr_R==-1, 0],Xtr_R_stan[Ytr_R==-1,1], label="x2", marker="o") # scatter plot the x2 values
plt.scatter(Xtr_R_stan[alpha > eps, 0],Xtr_R_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("Rings Kernel 1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.savefig("part_b_R_K1.png")
plt.show()

getConfusionMatrixAndMargin(Xtr_R_stan, Ytr_R,Xte_R_stan,Yte_R,kernel1,alpha,bias)
plotWaAndMisclass(history, "part_b_R_K1_Wa_Misclass.png", "Rings Dataset Kernel 1")

print("\nDataset = rings Kernel = K2")

alpha, bias, history = SMO(Xtr_R_stan,Ytr_R,kernel2)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_R_stan, Ytr_R, kernel2, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_R_stan[Ytr_R==1, 0],Xtr_R_stan[Ytr_R==1,1], label="x1", marker="o") # scatter plot the x1 values
plt.scatter(Xtr_R_stan[Ytr_R==-1, 0],Xtr_R_stan[Ytr_R==-1,1], label="x2", marker="o") # scatter plot the x2 values
plt.scatter(Xtr_R_stan[alpha > eps, 0],Xtr_R_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("Rings Kernel 2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.savefig("part_b_R_K2.png")
plt.show()

getConfusionMatrixAndMargin(Xtr_R_stan, Ytr_R,Xte_R_stan,Yte_R,kernel2,alpha,bias)
plotWaAndMisclass(history, "part_b_R_K2_Wa_Misclass.png", "Rings Dataset Kernel 2")


print("\nXOR Kernel = K1")

alpha, bias, history = SMO(Xtr_XOR_stan,Ytr_XOR,kernel1)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_XOR_stan, Ytr_XOR, kernel1, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_XOR_stan[Ytr_XOR==1, 0],Xtr_XOR_stan[Ytr_XOR==1,1], label="x1", marker="o") # scatter plot the x1 values
plt.scatter(Xtr_XOR_stan[Ytr_XOR==-1, 0],Xtr_XOR_stan[Ytr_XOR==-1,1], label="x2", marker="o") # scatter plot the x2 values
plt.scatter(Xtr_XOR_stan[alpha > eps, 0],Xtr_XOR_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("XOR Kernel 1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.savefig("part_b_XOR_K1.png")
plt.show()

getConfusionMatrixAndMargin(Xtr_XOR_stan, Ytr_XOR,Xte_XOR_stan,Yte_XOR,kernel1,alpha,bias)
plotWaAndMisclass(history, "part_b_XOR_K1_Wa_Misclass.png", "XOR Dataset Kernel 1")

print("\nXOR Kernel = K2")

alpha, bias, history = SMO(Xtr_XOR_stan,Ytr_XOR,kernel2)
print(f"bias = {bias}")
nonzero_alphas = alpha[alpha > 0]
print(f"# of nonzero alphas = {len(nonzero_alphas)}") # used for part E.3
eps = 1e-8

# get the grid desicion function values from SMO
xx, yy, fx = getGridAndDecisionFunction(Xtr_XOR_stan, Ytr_XOR, kernel2, alpha, bias)

fx = np.array(fx)
fx = fx.reshape(xx.shape) # make decision function 2D like xx and yy

plt.contour(xx,yy,fx,levels=[-1,0,1], linestyles=['--','-','--'], colors="black") # boundary fx=0 and +- 1

plt.scatter(Xtr_XOR_stan[Ytr_XOR==1, 0],Xtr_XOR_stan[Ytr_XOR==1,1], label="x1", marker="o") # scatter plot the x1 values
plt.scatter(Xtr_XOR_stan[Ytr_XOR==-1, 0],Xtr_XOR_stan[Ytr_XOR==-1,1], label="x2", marker="o") # scatter plot the x2 values
plt.scatter(Xtr_XOR_stan[alpha > eps, 0],Xtr_XOR_stan[alpha > eps, 1], label="Support Vectors", s=360, facecolors="none",
    edgecolors="red", linewidths=1) # scatter plot the support vector values alpha > epsilon make them small red see through circles 

plt.title("XOR Kernel 2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.savefig("part_b_XOR_K2.png")
plt.show()

getConfusionMatrixAndMargin(Xtr_XOR_stan, Ytr_XOR,Xte_XOR_stan,Yte_XOR,kernel2,alpha,bias)
plotWaAndMisclass(history, "part_b_XOR_K2_Wa_Misclass.png", "XOR Dataset Kernel 2")



# part c1 which is compare 10 random initializations
print("\n--- part c1: random initializations ---")

alphas = []
for seed in range(10):
    alpha, bias, history = SMO(
        Xtr_ISV_stan,
        Ytr_ISV,
        kernel1,
        T=10000,
        seed=seed
    )
    alphas.append(alpha)

    w = (alpha * Ytr_ISV) @ Xtr_ISV_stan

    print(
        f"seed {seed}: "
        f"w={np.round(w, 4)}, "
        f"b={bias:.4f}, "
        f"sv={np.sum(alpha > 1e-8)}"
    )

print("Are all alphas the same?")
alphas = np.array(alphas)
print(np.all(alphas == alphas[0]))

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
