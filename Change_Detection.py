# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:59:05 2025

@author: user
@studentID: 5885221
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sympy import symbols
from math import isclose
import cvxpy as cp
from collections import Counter
import scipy.io
np.random.seed(19680806)


""" Variables """
change_Dataset = scipy.io.loadmat('change_detection.mat')
a = np.array((change_Dataset['a']))
b = np.array((change_Dataset['b']))
c = np.array((change_Dataset['c']))
y = np.array((change_Dataset['y'])).reshape(-1, 1)
N = len(y)
#print(change_Dataset.keys())
print("y_values:",y)
variance = 0.5
v = np.random.normal(0, variance**2, (N, 1))
x = np.zeros((3, N))
w_true = np.vstack((a.T, b.T, c.T))   # shape (3, N)
w_single = np.zeros((3, N))
w_double = np.zeros((3, N))
w_prev = np.zeros((3, 1))

# optimized values of a,b,c assuming no infrequent switching, not solvable cause 3 unknowns and 1 equation:
""""
for i in range(6,N-3):
    x[i,:] = np.array(([y[i-1],y[i-2],y[i-3]]))
    test = x[i,:].T@x[i,:]
    print(test)
    print(x[i])
    w[i] = np.linalg.inv(x[i]@x[i].T)@x[i].T*y[i]
    print(w[i])
"""

""" Functions """
def convexSolve_single(x,y,W_prev):
    W = cp.Variable((3,1))
    Lapda = cp.Parameter(nonneg=True)
    Lapda.value = 0.5
    #x_shapeFixed = x.reshape((3,1))
    #constraints = []
    # box constraints:
    constraints = [W<=np.ones((3,1)), W>=-np.ones((3,1))]
    prob = cp.Problem(cp.Minimize(cp.square(y-x.T@W)+Lapda*(cp.norm(W-W_prev,2))), constraints)
    prob.solve(solver=cp.SCS)
    return W.value

def convexSolve_all(x,y,N,Lapda_value,var):
    W = cp.Variable((3,N))
    Lapda = cp.Parameter(nonneg=True)
    # should be tested with different lapda values!
    Lapda.value = Lapda_value
   
    # shapes:
    # X is 3,N
    # y is N,1
    # W is 3,N
    constraints = []
    obj = 0
    # box constraints:
    for i in range(1,N):
        obj += (cp.square(y[i]-x[:,i].T@W[:,i]))+Lapda*cp.norm(W[:,i]-W[:,i-1],2)
        #constraints.append(cp.norm(W[:,i],2)<=np.ones((3,1)))
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    return W.value

def subGradientDescent(x,y,N,W,alpha=1, lapda = 5):
    # need the dual function?
    ## probably not needed unless there is a distributed problem!
    # need to find subgradient, this is needed due to the norm in the objective function:
    ## comes from ((y[i]-x[:,i].T@W[:,i])**2)'
    gradientF0 = 0-2*x[:,i]*y[i] + 2 * x[:,i]@x[:,i]@W[:,i] # need to write this out better!
    v = lapda*(W[:,i]-W[:,i-1])
    gradient_changeCorrection = v/np.norm(v)
    subgradient = gradientF0 + gradient_changeCorrection

    # take steps (optimal alpha comes later):
    W_descent = W_descent - alpha*subgradient
    return W
    


""" Main Code """
""" Question 1: Creation of the optimization problem """
for i in range(3,N-3):
    x_slice = np.vstack((y[i-1],y[i-2],y[i-3]))
    x[:,i] = x_slice.squeeze()
    w_single[:,i] = convexSolve_single(x_slice,y[i],w_prev).squeeze()
    w_prev = w_single[:,i]

lapda = 5
w_double = convexSolve_all(x,y,N,lapda,variance)

""" Plotting """

# Estimated versus true values (single):
plt.figure(figsize=(9, 5))

plt.plot(w_single[0,:], label="Estimated a", linewidth=2)
plt.plot(w_single[1,:], label="Estimated b", linewidth=2)
plt.plot(w_single[2,:], label="Estimated c", linewidth=2)

plt.plot(w_true[0,:], '--', label="True a")
plt.plot(w_true[1,:], '--', label="True b")
plt.plot(w_true[2,:], '--', label="True c")

plt.xlabel("Time index t")
plt.ylabel("Coefficient value")
plt.title("Piecewise-constant AR coefficient estimation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Error values (single):
err_a = (w_single[:,0] - w_true[:,0])**2
err_b = (w_single[:,1] - w_true[:,1])**2
err_c = (w_single[:,2] - w_true[:,2])**2

plt.figure(figsize=(9, 5))

plt.plot(err_a, label="Squared error a", linewidth=2)
plt.plot(err_b, label="Squared error b", linewidth=2)
plt.plot(err_c, label="Squared error c", linewidth=2)

plt.xlabel("Time index t")
plt.ylabel("Squared error")
plt.title("Coefficient estimation error over time")
plt.yscale('log',base=10) 
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Estimated versus true values (all):
plt.figure(figsize=(10,5))

# Estimated weights
plt.plot(w_double[0,:], label='Estimated a', linewidth=2)
plt.plot(w_double[1,:], label='Estimated b', linewidth=2)
plt.plot(w_double[2,:], label='Estimated c', linewidth=2)

# True weights
plt.plot(w_true[0,:], '--', label='True a')
plt.plot(w_true[1,:], '--', label='True b')
plt.plot(w_true[2,:], '--', label='True c')

plt.xlabel('Time index')
plt.ylabel('Coefficient value')
plt.title('Batch estimation of piecewise-constant AR coefficients')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()