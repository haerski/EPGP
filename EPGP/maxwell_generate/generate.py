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
import h5py
import time as t

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)
torch.manual_seed(13)


# Set grid
axis = torch.linspace(-2,2,11)

## PDE solving Gaussian Process
# Input: a lM * 4 tensor of Monte-Carlo points
# Output: a lM * ll * 6 tensor of Noetherian operators evaluated at MC points
def noethOp(M, op):
    x,y,z,t = M.unbind(-1)
    O = torch.zeros_like(x)
    if op == 0:
        return torch.stack([x*z, y*z, z**2-t**2, y*t, -x*t, O], -1)
    if op == 1:
        return torch.stack([x*y, y**2-t**2, y*z, -z*t, O, x*t], -1)
    if op == 2:
        return torch.stack([-y**2-z**2, x*y, x*z, O, z*t, -y*t], -1)
    if op == 3:
        return torch.stack([-y*t, x*t, O, x*z, y*z, z**2-t**2], -1)
    if op == 4:
        return torch.stack([z*t, O, -x*t, x*y, y**2-t**2, y*z], -1)
    if op == 5:
        return torch.stack([O, -z*t, y*t, -y**2-z**2, x*y, x*z], -1)
    else:
        raise Exception("not a valid operator")

def getVarPoints(diracs):
    x, y, z = diracs.unbind(1)
    t = (x**2 + y**2 + z**2).sqrt()
    return torch.stack( [ torch.stack([x,y,z,t],1), torch.stack([x,y,z,-t],1) ] )


def Phi(Z, X):
    cols = torch.stack([ noethOp(Mp, i) for i, Mp in enumerate(Z.unbind(1)) ], 1)

    rows = (Z.unsqueeze(-2) * X).sum(-1).exp()

    summands = rows.unsqueeze(-1) * cols.unsqueeze(-2)

    return summands.flatten(-2,-1).flatten(1,2).mean(0)



# Initial data, make a spiral in electric field
t_init = torch.linspace(0,0.5,11)
Ps = torch.cartesian_prod(axis,axis,axis,t_init)
mask1 = Ps[:,2].abs() <= 1e-5
mask2 = ((Ps[:,0].square() + Ps[:,1].square()).sqrt() - 1).abs() < 0.5
mask = mask1 & mask2
X = Ps[mask]
Y = torch.stack([-X[:,1], X[:,0], X[:,2]],-1).view(-1,1)


# Inference
n_spectral = 32
n_ops = 6
base_pts = torch.randn((n_ops, n_spectral, 3), device=device)
eps = torch.tensor(np.log(1e-6), device=device).requires_grad_()
S_diag = torch.full((n_spectral*n_ops,), 0., device=device)
opt = torch.optim.Adam([base_pts, eps, S_diag], lr = 1e-4)


Z = getVarPoints(base_pts.flatten(0,1)).unflatten(1, (n_ops, n_spectral))
PhiX = Phi(Z * 1.j, X)
# Remove magn field
PhiX = PhiX.unflatten(-1,(-1,6))[:,:,:3].flatten(-2,-1)
A = (n_ops * n_spectral) * torch.diag_embed((eps - S_diag).exp()) + PhiX @ PhiX.H
LA = torch.linalg.cholesky(A)
alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)

nlml = 1/(2*eps.exp()) * (Y.norm().square() - alpha.norm().square())
nlml += (PhiX.shape[1] - PhiX.shape[0])/2 * eps
nlml += LA.diag().real.log().sum()
nlml += 0.5*S_diag.sum()

opt.zero_grad()
nlml.backward()
opt.step()

with torch.no_grad():
    train_pred = PhiX.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)
    err = (train_pred.real - Y).square().mean().sqrt()
    print(26*"~" + f'\nepoch {epoch}\n\
nlml {nlml}\n\
err {err}\n\
eps {eps.exp()}\n')

# predict

t_predict = torch.linspace(0,8,300)
Ps = torch.cartesian_prod(axis,axis,axis,t_predict)
Z = getVarPoints(base_pts.flatten(0,1)).unflatten(1, (n_ops, n_spectral))
Phi_ = Phi(Z * 1.j, Ps).to(device)
A = (n_ops * n_spectral) * torch.diag_embed((eps - S_diag).exp()) + PhiX @ PhiX.H
LA = torch.linalg.cholesky(A)
alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)
pred = Phi_.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)

time = t_predict

pred.real.flatten().detach().cpu().numpy().tofile("generate.dat")
axis.cpu().numpy().tofile("axis.dat")
time.cpu().numpy().tofile("time.dat")