## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)
torch.manual_seed(13)



# Sampling parameters etc
n_axis = 71
n_time = 81

axis = torch.linspace(-3.5,3.5,n_axis, device=device)
time = torch.linspace(0,0.3,n_time, device=device)
Ps = torch.cartesian_prod(axis,axis,time)
# Number of points
lP = Ps.shape[0]

# Heat EPGP kernel
def k(X,XX, sigma):
    x,y,t = X.split(1,1)
    xx,yy,tt = ( v.T for v in XX.split(1,1) )
    denom = 1/sigma +(2*(t+tt))
    return torch.exp(-((x-xx).square()+(y-yy).square()) / (2*denom) ) / denom


def posterior_mean(X,Y,sigma):
    Y = Y.view(-1,1)

    kXX = k(X,X,sigma)
    k_X = k(Ps,X,sigma)

    eps = 1e-6
    A = kXX + eps * torch.eye(kXX.shape[0], device=device)
    L = torch.linalg.cholesky(A)
    alpha = torch.linalg.solve_triangular(L, Y, upper=False)
    alpha1 = torch.linalg.solve_triangular(L.T, alpha, upper=True)

    return k_X @ alpha1

# Generate initial data
data_axis = torch.linspace(-5,5,101, device=device)
data_time = torch.linspace(0,0.,1, device=device)
data_Ps = torch.cartesian_prod(data_axis,data_axis,data_time)

mask = data_Ps[:,2].abs() < 0.001
X = data_Ps[mask]
Y = torch.where((X[:,:2] - torch.tensor([1.5, -1.5])).square().sum(1) <= 0.25, 1, 0)
Y += torch.where((X[:,:2] - torch.tensor([1.5, 1.5])).square().sum(1) <= 0.25, 1, 0)

Y += torch.where(
    (((0.5*X[:,1].square() - 2) - X[:,0]).square() < 0.1) &
    (X[:,1].abs() <= 2), 1, 0)

# sn.heatmap(Y.view(101,101).flip(0), cbar=None, yticklabels=False, xticklabels=False)
# plt.savefig("initial_heat.png")

# Posterior
sol = posterior_mean(X,Y,20)
sol.detach().numpy().tofile("sol20.dat")
sol = posterior_mean(X,Y,2)
sol.detach().numpy().tofile("sol2.dat")
axis.numpy().tofile("axis.dat")
time.numpy().tofile("time.dat")