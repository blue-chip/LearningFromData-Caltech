#!/usr/bin/env python

"""
This script creates a support vector machine (SVM) implementation in Python
using the CVXOPT module. This module computes the quadratic optimization
needed to find the support vectors. The domain is restricted to two
dimensions ranging from -1 to 1. The visualizations are done with the
Seaborn module.
"""


import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import seaborn


# Creates the target function

data_range = [-1, 1]  # Data domain is a square with this range for each side

pt1_targetf = np.random.uniform(
    data_range[0],
    data_range[1],
    2)
pt2_targetf = np.random.uniform(
    data_range[0],
    data_range[1],
    2)

slope = ((pt2_targetf[1]-pt1_targetf[1])
         /
         (pt2_targetf[0]-pt1_targetf[0]))
y_intercept = pt1_targetf[1] - slope * pt1_targetf[0]

leftmostpt_targetf = [data_range[0], slope*(data_range[0])+y_intercept]
rightmostpt_targetf = [data_range[1], slope*(data_range[1])+y_intercept]


# Plots the target function

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.plot(
    [leftmostpt_targetf[0], rightmostpt_targetf[0]],
    [leftmostpt_targetf[1], rightmostpt_targetf[1]])
plt.axis((
    data_range[0],
    data_range[1],
    data_range[0],
    data_range[1]))
plt.ion()
plt.show()
plt.pause(0.001)


# Creates N training points, assigns them a label and plots them

N = 20

x_values = np.random.uniform(
    data_range[0],
    data_range[1],
    N)
y_values = np.random.uniform(
    data_range[0],
    data_range[1],
    N)

labels = []


# Computing the labels (class or category) from the target function
for index, each in enumerate(x_values):
    y_value = slope*(each) + y_intercept

    if y_values[index] < y_value:
        labels.append(-1.0)
    else:
        labels.append(1.0)

# Plotting the training points
plt.scatter(
    x_values,
    y_values,
    c=labels,
    cmap='bwr',
    alpha=1,
    s=100,
    edgecolors='k'
    )

# Combining the data vectors into an X matrix of N vectors of features
X_svm = np.column_stack((
    x_values,
    y_values
    ))
# Changing the class list into a Numpy array
labels = np.array(labels)
# Initializing the N x N Gram matrix
K = np.zeros(
    shape=(
        len(x_values),
        len(x_values)
        )
    )

# Computing the inner products for each pair of vectors
for i in range(len(x_values)):
    for j in range(len(x_values)):
        K[i, j] = np.dot(X_svm[i], X_svm[j])

# Generating all the matrices and vectors
P = cvxopt.matrix(np.outer(labels, labels) * K)
q = cvxopt.matrix(np.ones(len(x_values)) * -1)
G = cvxopt.matrix(np.vstack([
    np.eye(len(x_values)) * -1,
    np.eye(len(x_values))
    ]))
h = cvxopt.matrix(np.hstack([
    np.zeros(len(x_values)),
    np.ones(len(x_values)) * 999999999.0
    ]))
A = cvxopt.matrix(labels, (1, len(x_values)))
b = cvxopt.matrix(0.0)

# Solving the QP problem
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# Display the Lagrange multipliers
a = np.ravel(solution['x'])
print a

# Create a boolean list of non-zero alphas
ssv = a > 1e-5
# Select the index of these alphas. They are the support vectors.
ind = np.arange(len(a))[ssv]
# Select the corresponding alphas a, support vectors sv and class labels sv_y
a = a[ssv]
sv = X_svm[ssv]
sv_y = labels[ssv]

# Plotting the support vectors
plt.scatter(
    sv[:, 0],
    sv[:, 1],
    facecolors='none',
    s=400,
    edgecolors='k'
    )

# Computing the weights w_svm
w_svm = np.zeros(X_svm.shape[1])

for each in range(len(a)):
    w_svm += a[each] * sv_y[each] * sv[each]
print 'w_svm:', w_svm

# Computing the intercept b_svm
b_svm = sv_y[0] - np.dot(w_svm, sv[0])
print 'b_svm:', b_svm

# Plot of SVM function
x2_lefttargeth = -(w_svm[0]*(-1)+b_svm)/w_svm[1]
x2_righttargeth = -(w_svm[0]*(1)+b_svm)/w_svm[1]

ax1.plot([-1, 1], [x2_lefttargeth, x2_righttargeth])
