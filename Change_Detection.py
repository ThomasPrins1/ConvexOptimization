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
import math
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
print(w_true.shape)
w_single = np.zeros((3, N))
w_double = np.zeros((3, N))
w_gradientDescent = np.zeros((3, N))
w_prev = np.zeros((3, 1))
iterMax = 3000
eps = 1e-8

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

def subGradientDescent(x,y,N,Lapda,Alpha=1e-5, tol=1e-10):
    # need the dual function?
    ## probably not needed unless there is a distributed problem!
    # need to find subgradient, this is needed due to the norm in the objective function:
    ## comes from ((y[i]-x[:,i].T@W[:,i])**2)'
    # normalize?
    x /= np.linalg.norm(x, axis=0, keepdims=True) + eps

    W_out = np.zeros((3, N-3))
    gradient_changeCorrection = np.zeros_like(W_out)
    gradientF0 = np.zeros_like(W_out)
    #print(((x@x.T@W_out)).shape)
    #print((x@y).shape)
     
    for iter,i in enumerate(range(iterMax)):
        Alpha /= np.sqrt(i+1) 
        W_old = W_out.copy()
        for j in range(N-3):
            grad_temp = np.zeros(3)
            #boundary conditions:
            if j <= 0:
               s_n = 0
            else:
               s_n = (W_out[:,j]-W_out[:,j-1])
            if j >= N-4:
                s_p = 0
            else:
                s_p = (W_out[:,j+1]-W_out[:,j])
            #print("s_n & s_p:")
            #print(s_n)
            #print(s_p)
            # check if norm isnt at 0 to avoid division by 0 (this results in subgradient):

            if np.linalg.norm(s_n) > eps:
                grad_temp += Lapda * (s_n / np.linalg.norm(s_n,2))

            if np.linalg.norm(s_p) > eps:
                grad_temp -= Lapda * (s_p / np.linalg.norm(s_p,2))
                
            gradient_changeCorrection[:,j] = grad_temp
            gradientF0[:,j] = -2 * x[:, j] * (y[j] - x[:, j].T @ W_out[:, j] ) # need to write this out better in report!

            # gradient clipping to reduce the amount of sudden change that can happen?
            grad_norm = np.linalg.norm(gradientF0[:, j])
            if grad_norm > 1:
                gradientF0[:, j] *= 1 / grad_norm
            # some box constraints?

            # print("gradients:")
            # print(x[:, j].T @ W_out[:, j])
            # print(gradient_changeCorrection[:,j])
            # print(gradientF0[:,j])
        subgradient = gradientF0 + gradient_changeCorrection
        W_out -= Alpha*subgradient
        rel_change = np.linalg.norm(W_out - W_old) / (np.linalg.norm(W_old) + eps)
        #if rel_change < tol:
        #    print("convergence at: ",iter)
        #    break
    return W_out
    


""" Main Code """
""" Question 1: Creation of the optimization problem """
for i in range(3,N-3):
    x_slice = np.vstack((y[i-1],y[i-2],y[i-3]))
    x[:,i] = x_slice.squeeze()
    w_single[:,i] = convexSolve_single(x_slice,y[i],w_prev).squeeze()
    w_prev = w_single[:,i]

lapda = 5
print(x.shape)
w_double = convexSolve_all(x,y,N,lapda,variance)

best_mse = 100000
lapda_space = np.linspace(3.5,4,5)
for l in lapda_space:
    w_temp = subGradientDescent(x,y,N,l,1)
    fix = np.hstack([np.zeros((3,3)),w_true])
    mse_w = np.mean((w_temp - w_true)**2)
    if mse_w < best_mse:
        best_mse = mse_w
        w_gradientDescent = w_temp
        print(l)
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

# Estimated versus true values (not CVX):
plt.figure(figsize=(10,5))

# Estimated weights
plt.plot(w_gradientDescent[0,:], label='Estimated a', linewidth=2)
plt.plot(w_gradientDescent[1,:], label='Estimated b', linewidth=2)
plt.plot(w_gradientDescent[2,:], label='Estimated c', linewidth=2)

# True weights
plt.plot(w_true[0,:], '--', label='True a')
plt.plot(w_true[1,:], '--', label='True b')
plt.plot(w_true[2,:], '--', label='True c')

plt.xlabel('Time index')
plt.ylabel('Coefficient value')
plt.title('Batch estimation of piecewise-constant AR coefficients (gradient descent)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()