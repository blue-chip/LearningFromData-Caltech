#!/usr/bin/env python

"""
This script creates a support vector machine (SVM) implementation in Python
using the CVXOPT module. This module computes the quadratic optimization
needed to find the support vectors. The domain is restricted to two
dimensions ranging from -1 to 1. The visualizations are done with the
Seaborn module.
"""

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def generate_2d_points(limits):
    return np.random.uniform(
        low=limits[0],
        high=limits[1],
        size=2
    )


def slope_intercept(point1, point2):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    y_intercept = point1[1] - slope * point1[0]
    return [slope, y_intercept]


# Create target function
data_range = [-1, 1]

pt1_targetf = generate_2d_points(data_range)
pt2_targetf = generate_2d_points(data_range)

slope, y_intercept = slope_intercept(pt1_targetf, pt2_targetf)

leftmostpt_targetf = [
    data_range[0],
    slope * (data_range[0]) + y_intercept
]
rightmostpt_targetf = [
    data_range[1],
    slope * (data_range[1]) + y_intercept
]

# Plots the target function
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
target_function, = plt.plot(
    [leftmostpt_targetf[0], rightmostpt_targetf[0]],
    [leftmostpt_targetf[1], rightmostpt_targetf[1]],
    label='Target function'
)
plt.axis((
    data_range[0],
    data_range[1],
    data_range[0],
    data_range[1]
))
plt.ion()
plt.show()

# Creates N training points, assigns them a label and plots them
N = 50

x_values = np.random.uniform(
    data_range[0],
    data_range[1],
    N
)
y_values = np.random.uniform(
    data_range[0],
    data_range[1],
    N
)

y_values_targetf = [slope * x + y_intercept for x in x_values]

labels = [-1.0 if y_values[i] < y else 1.0 for i, y in enumerate(y_values_targetf)]

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
K = np.dot(X_svm, np.transpose(X_svm))

# Generating all the matrices and vectors
P = cvxopt.matrix(np.outer(labels, labels) * K)
q = cvxopt.matrix(np.ones(len(x_values)) * -1)
G = cvxopt.matrix(np.vstack(
    [
        np.eye(len(x_values)) * -1,
        np.eye(len(x_values))
    ]
))
h = cvxopt.matrix(np.hstack(
    [
        np.zeros(len(x_values)),
        np.ones(len(x_values)) * 999999999.0
    ]
))
A = cvxopt.matrix(labels, (1, len(x_values)))
b = cvxopt.matrix(0.0)

# Solving the QP problem
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# Display the Lagrange multipliers
a = np.ravel(solution['x'])
print(a)

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
print('w_svm:', w_svm)

# Computing the intercept b_svm
b_svm = sv_y[0] - np.dot(w_svm, sv[0])
print('b_svm:', b_svm)

# Plot of SVM function
x2_lefttargeth = -(w_svm[0] * (-1) + b_svm) / w_svm[1]
x2_righttargeth = -(w_svm[0] * (1) + b_svm) / w_svm[1]

svm_function, = ax1.plot(
    [-1, 1],
    [x2_lefttargeth, x2_righttargeth],
    label='SVM function'
)

plt.legend(
    handles=[
        target_function,
        svm_function
    ]
)
