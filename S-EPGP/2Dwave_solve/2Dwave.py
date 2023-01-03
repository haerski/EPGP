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
n_axis = 51
n_time = 131

axis = torch.linspace(-2,2,n_axis, device=device)
time = torch.linspace(0,4,n_time, device=device)
Ps = torch.cartesian_prod(axis,axis,time)
# Number of points
lP = Ps.shape[0]


# Initial dataset
data_axis = torch.linspace(-5,5, 101, device=device)
data_time = torch.linspace(0,0,1, device=device)
data_Ps = torch.cartesian_prod(data_axis,data_axis,time)

mask = data_Ps[:,2] == 0.
X = data_Ps[mask]

Y = torch.where( ((X[:,0]-1).abs() < 1e-1) & (X[:,1].abs() < 1), 1., 0 )
Y += torch.where( ((X[:,0] - X[:,1] + 2).abs() < 1e-1) & (X[:,1] >= 0.2) & (X[:,1] <= 1.8) , 1., 0 )
Y = Y.view(-1,1)

X = X.to(torch.complex128)
Y = Y.to(torch.complex128)

# sn.heatmap(Y.view(n_axis, n_axis))

def getVarietyPoints(base):
    x,y = base.unbind(1)
    t = torch.sqrt(x.square() + y.square())

    return torch.stack([ torch.stack([x,y,t],1), torch.stack([x,y,-t],1) ])

def Phi(base, X):
    pts = getVarietyPoints(base)
    # return (pts.inner(X) * 1.j).exp().mean(0)
    return (pts.inner(X)).exp().mean(0)

def train(N):
    for epoch in range(N):
        PhiX = Phi(MC_base * 1.j, X)
        A = torch.diag_embed((eps - S_diag).exp()) + PhiX @ PhiX.H
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
eps {eps.exp()}\n\
base std {MC_base.std(0)}\n\
min,max {train_pred.real.min().detach(),train_pred.real.max().detach()}')


n_MC = 2000
# MC_axis = torch.linspace(-1,1, n_MC, device=device) * 30
MC_base = (torch.randn((n_MC, 2), device=device)).requires_grad_()
# MC_base = torch.cartesian_prod(MC_axis,MC_axis).requires_grad_()
S_diag = torch.full((n_MC,), -np.log(n_MC), requires_grad=False, device=device)
# S_diag = torch.full((n_MC**2,), -np.log(n_MC**2), requires_grad=False, device=device)
eps = torch.tensor(np.log(1e-2), requires_grad=True, device=device)

opt = torch.optim.Adam([
    {'params': MC_base, 'lr': 1e-1},
    {'params': eps, 'lr': 1e-2}])
train(100000)
opt = torch.optim.Adam([
    {'params': MC_base, 'lr': 1e-1},
    {'params': eps, 'lr': 1e-2}])
train(100000)
opt = torch.optim.Adam([
    {'params': MC_base, 'lr': 1e-2},
    {'params': [S_diag, eps], 'lr': 1e-2}])
train(1000)
opt = torch.optim.Adam([
    {'params': MC_base, 'lr': 1e-3},
    {'params': [S_diag, eps], 'lr': 1e-3}])
train(300)


torch.save({
            'MC_base': MC_base.cpu(),
            'S_diag': S_diag.cpu(),
            'eps': eps.cpu(),
    }, "state.pt")


st = torch.load("state.pt")
MC_base = st['MC_base']
S_diag = st['S_diag']
eps = st['eps']

# Prediction
Phi_ = Phi(MC_base * 1.j, Ps.to(torch.complex128)).to(device)
PhiX = Phi(MC_base * 1.j, X)
A = torch.diag_embed((eps - S_diag).exp()) + PhiX @ PhiX.H
LA = torch.linalg.cholesky(A)
alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)
pred = Phi_.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)
pred = pred.real

pred.detach().cpu().numpy().tofile("pred.dat")
axis.cpu().numpy().tofile("axis.dat")
time.cpu().numpy().tofile("time.dat")


plt.ion()
f, ax = plt.subplots()
# sn.kdeplot(x = MC_base.detach().numpy()[:,0], y = MC_base.detach().numpy()[:,1], fill=True)
sn.scatterplot(x = MC_base.detach().numpy()[:,0], y = MC_base.detach().numpy()[:,1], s=10)
sn.kdeplot(x = MC_base.detach().numpy()[:,0], y = MC_base.detach().numpy()[:,1], bw_adjust=0.5, fill=True)