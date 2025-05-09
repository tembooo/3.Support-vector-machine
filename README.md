# 3.Support-vector-machine
![image](https://github.com/user-attachments/assets/be5b472e-7fe5-40e6-841f-09b80dcdfc36)

## 🤖 Linear Support Vector Machine (SVM) Implementation

In this task, we implement a linear support vector machine (SVM) in either Matlab or Python.  
The goal is to define a function that receives the training data and class labels and returns the classifier parameters: weight vector and bias.

The dataset is already prepared, and some of the samples are designated as support vectors for each class.

📁 Additional data files: CSV and MAT formats are provided.

---

### 💡 Hints

---

### 1. Determining the classifier parameters

First, we define an objective function that needs to be maximized under certain constraints.  
After solving the optimization, we calculate the weight vector and the bias value using the results.  
For computing the bias, you can select any support vector from the first class.

---

### 2. Solving the optimization problem

We use quadratic programming to solve the dual form of the SVM.

#### 🧪 Using Matlab:

Matlab provides a built-in function in the Optimization Toolbox to solve this type of problem.  
You need to define the matrix terms related to the training data and class labels, and pass them to the optimization function.

Hints:
- Store your training samples and class labels in matrix form.
- Define the necessary constraints: equality constraint and bounds.
- Some variables like inequality constraints can be left empty if not needed.

---

#### 🐍 Using Python:

In Python, we use the `cvxopt` library.  
First, convert all numpy arrays into the format expected by the solver.  
Make sure all data is in float format.  
Then use the solver function by passing the necessary arguments.

To apply the bounds, create matrices for lower and upper limits using helper functions from numpy and `cvxopt`.

---

### 3. Implementation Steps

1. Extract the class information from the input.  
2. Prepare all matrices and vectors needed for the optimization step.  
3. Use a suitable quadratic programming method to solve for the multipliers.  
4. From the results, compute the final model parameters: the weight vector and the bias.

---

📌 Tip: If you're using Python, install `cvxopt` via pip.  
📌 Tip: In Matlab, make sure the Optimization Toolbox is available and loaded.



```python
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
# Load the data
file_path = 'C:\\12.LUT\\00.Termic Cources\\2.pattern recognition\\jalase5\\Code\\t103.csv'
# Change this to the correct path
data = pd.read_csv(file_path, header=None)

# Step 1: Preprocessing the data
X = data.iloc[:, :2].values  # Features (first two columns)
y = data.iloc[:, 2].values  # Labels (third column)
# Convert class labels to -1 and 1 for SVM
y = np.where(y == 2, -1, 1)  # Assuming class "2" is recoded as -1 and any other as +1
# Step 2: Define the quadratic programming problem
# Number of data points and features
m, n = X.shape
# Construct the matrices for the quadratic programming problem
K = np.dot(X, X.T)  # Kernel (linear kernel)
P = matrix(np.outer(y, y) * K)  # P = y*y^T * K
q = matrix(-np.ones((m, 1)))  # q = -1 (m-length vector)
# G and h represent the inequality constraints
G = matrix(-np.eye(m))  # G = -I (identity matrix)
h = matrix(np.zeros(m))  # h = 0 (m-length vector)
# A and b represent the equality constraint
A = matrix(y, (1, m), 'd')  # A = y^T
b = matrix(0.0)  # b = 0
# Step 3: Solve the quadratic programming problem using cvxopt
sol = solvers.qp(P, q, G, h, A, b)
# Extract the Lagrange multipliers (alphas)
alphas = np.array(sol['x']).flatten()
# Step 4: Calculate w and w0 (bias term)
w = np.sum(alphas[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
# Support vectors have non-zero Lagrange multipliers
support_vector_indices = np.where(alphas > 1e-5)[0]
w0 = np.mean([y[i] - np.dot(w, X[i]) for i in support_vector_indices])
# Now, let's count the number of support vectors for each class
support_vector_labels = y[support_vector_indices]
# Count the number of support vectors for each class
num_class1_support_vectors = np.sum(support_vector_labels == 1)
num_class2_support_vectors = np.sum(support_vector_labels == -1)
num_class1_support_vectors, num_class2_support_vectors
# Display results
print("Weight vector (w):", w)
print("Bias (w0):", w0)
print("Support vector indices:", support_vector_indices)
# Retrieve the labels of the support vectors
support_vector_labels = y[support_vector_indices]
# Count the number of support vectors for each class
num_class1_support_vectors = np.sum(support_vector_labels == 1)
num_class2_support_vectors = np.sum(support_vector_labels == -1)
print(f"Number of support vectors for Class 1 (+1): {num_class1_support_vectors}")
print(f"Number of support vectors for Class 2 (-1): {num_class2_support_vectors}")
```
![image](https://github.com/user-attachments/assets/70ceadd1-d86e-4dff-bafd-9a1098d248a9)
