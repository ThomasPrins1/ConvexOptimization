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
variance = 0.5
v = np.random.normal(0, variance**2, (N, 1))
x = np.zeros((3, N))
w_true = np.vstack((a.T, b.T, c.T))   # shape (3, N)
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

def convexSolve_all(x,y,N,Lapda_value):
    #x /= (np.linalg.norm(x, axis=0, keepdims=True) + eps)
    W = cp.Variable((3,N-3))
    Lapda = cp.Parameter(nonneg=True)
    # should be tested with different lapda values!
    Lapda.value = Lapda_value
    const = cp.Constant(0)
    obj = cp.Constant(0)
    # shapes:
    # X is 3,N
    # y is N,1
    # W is 3,N

    for i in range(3,N):
        obj += (cp.square(y[i]-x[:,i].T@W[:,i-3]))
        if i>4:
            obj += Lapda*(cp.norm(W[:,i-3]-W[:,i-4],2))
            const += (cp.norm(W[:,i-3]-W[:,i-4],1))
    constraints = [const<=3+eps]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    return W.value

def project_TV_ball(W, tau):
    # Compute differences
    D = W[:, 1:] - W[:, :-1]        # 3 x (T-1)
    norms = np.linalg.norm(D, axis=0)
    tau = 0.3*np.sum(np.linalg.norm(W[:,1:] - W[:,:-1], axis=0))
    if norms.sum() <= tau:
        return W

    # Find theta via sorting
    sorted_norms = np.sort(norms)[::-1]
    cumsum = np.cumsum(sorted_norms)
    rho = np.where(sorted_norms * np.arange(1, len(norms)+1) > (cumsum - tau))[0][-1]
    theta = (cumsum[rho] - tau) / (rho + 1)

    # Shrink
    scale = np.maximum(1 - theta / (norms + eps), 0)
    D_proj = D * scale

    # Reconstruct W
    W_proj = np.zeros_like(W)
    for j in range(1, W.shape[1]):
        W_proj[:, j] = W_proj[:, j-1] + D_proj[:, j-1]

    return W_proj

def projectedSubGradientDescent(x,y,N,Lapda,Alpha=1e-0, tol=1e-6):
    # need the dual function?
    ## probably not needed unless there is a distributed problem!
    # need to find subgradient, this is needed due to the norm in the objective function:
    ## comes from ((y[i]-x[:,i].T@W[:,i])**2)'
    # normalize?
    x_trim = x[:, 3:]   # 3 x (N-3)
    y_trim = y[3:]      # length (N-3)
    x_norm = x_trim / (np.linalg.norm(x_trim, axis=0, keepdims=True) + eps)
    W_out = np.zeros((3, N-3))
    gradient_F0 = np.zeros_like(W_out)
    for k in range(iterMax):
        print(k)
        Alpha_value = Alpha/np.sqrt(k+1) 
        gradient_F1 = np.zeros_like(W_out)
        for j in range(N-3):
            gradientTemp1 = 2*x_norm[:,j]*y_trim[j]
            gradientTemp0 = 2*x_norm[:,j]*(x_norm[:,j].T@W_out[:,j])
            gradient_F0[:,j] = gradientTemp0 - gradientTemp1
        for j in range(1,N-3):
            change = W_out[:,j]-W_out[:,j-1]
            change_norm = np.linalg.norm(change,2)
            if change_norm > eps:
                g = Lapda * change / change_norm
                gradient_F1[:, j] += g
                gradient_F1[:, j-1] -= g
        subgradient = gradient_F0 + gradient_F1
        W_temp = W_out - Alpha_value*subgradient
        W_temp2 = project_TV_ball(W_temp, 1)
        print(np.linalg.norm(W_out))
        if np.linalg.norm(subgradient)<tol:
            print("convergence at: ",k)
            break
        W_out = W_temp2
    return W_out

def subGradientDescent(x,y,N,Lapda,Alpha=1e-2, tol=1e-4):
    # need the dual function?
    ## probably not needed unless there is a distributed problem!
    # need to find subgradient, this is needed due to the norm in the objective function:
    ## comes from ((y[i]-x[:,i].T@W[:,i])**2)'
    # normalize?
    x_trim = x[:, 3:]   # 3 x (N-3)
    y_trim = y[3:]      # length (N-3)
    x_norm = x_trim / (np.linalg.norm(x_trim, axis=0, keepdims=True) + eps)
    W_out = np.zeros((3, N-3))
    gradient_F0 = np.zeros_like(W_out)
    for k in range(iterMax):
        print(k)
        Alpha_value = Alpha/np.sqrt(k+1) 
        gradient_F1 = np.zeros_like(W_out)
        for j in range(N-3):
            gradientTemp1 = 2*x_norm[:,j]*y_trim[j]
            gradientTemp0 = 2*x_norm[:,j]*(x_norm[:,j].T@W_out[:,j])
            gradient_F0[:,j] = gradientTemp0 - gradientTemp1
        #print(gradient_F0)
        for j in range(1,N-3):
            change = W_out[:,j]-W_out[:,j-1]
            change_norm = np.linalg.norm(change,2)
            if change_norm > eps:
                g = Lapda * change / change_norm
                gradient_F1[:, j] += g
                gradient_F1[:, j-1] -= g
        subgradient = gradient_F0 + gradient_F1
        W_out -= Alpha_value*subgradient
        print(np.linalg.norm(W_out))
        if np.linalg.norm(subgradient)<tol:
            print("convergence at: ",k)
            break
    return W_out
"""""
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
"""""
    
    


""" Main Code """
""" Question 1: Creation of the optimization problem """
for i in range(3,N-3):
    x_slice = np.vstack((y[i-1],y[i-2],y[i-3]))
    x[:,i] = x_slice.squeeze()

best_mse1 = np.inf
lapda_space = np.linspace(0,2,6)
for l in lapda_space:
    w_temp_double = convexSolve_all(x,y,N,l)
    mse_w_double = np.mean((w_temp_double - w_true)**2)
    if mse_w_double < best_mse1:
        best_mse1 = mse_w_double
        w_double = w_temp_double
        print(l)

best_mse2 = np.inf
lapda_space = np.linspace(0,10,6)
for l in lapda_space:
    w_temp_GD = projectedSubGradientDescent(x,y,N,l)
    #fix = np.hstack([np.zeros((3,3)),w_true])
    mse_w_GD = np.mean((w_temp_GD - w_true)**2)
    if mse_w_GD < best_mse2:
        best_mse2 = mse_w_GD
        w_gradientDescent = w_temp_GD
        print(l)
""" Plotting """

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