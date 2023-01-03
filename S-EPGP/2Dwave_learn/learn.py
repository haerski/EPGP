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
from copy import deepcopy

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)
torch.manual_seed(13)



# Sampling using Mathematica
axis = torch.tensor(h5py.File('axis.h5')['Dataset1'][...], dtype=torch.float64)
time = torch.tensor(h5py.File('ts.h5')['Dataset1'][...], dtype=torch.float64)

n_axis = axis.shape[0]
n_time = time.shape[0]

Ps = torch.cartesian_prod(axis,axis,time)
xx, yy, tt = torch.meshgrid((axis, axis, time), indexing="ij")
# Number of points
lP = Ps.shape[0]
# Output dimension
ll = 1

# A numerical solution
udata = torch.tensor(h5py.File('wave.h5')['Dataset1'][...], dtype=torch.float64)
udata = udata.squeeze()

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


n_MC = 16
MC_base = (torch.randn((n_MC, 2), device=device)).requires_grad_()
S_diag = torch.full((n_MC,), -np.log(n_MC), requires_grad=True, device=device)
eps = torch.tensor(np.log(1e-2), requires_grad=True, device=device)

# Initial data
mask = Ps[:,2] <= time[2]
X = Ps[mask].to(torch.complex128).to(device)
Y = udata.flatten()[mask].view(-1,1).to(device)

opt = torch.optim.Adam([
    {'params': MC_base, 'lr': 1e-1},
    {'params': S_diag, 'lr': 1e-1},
    {'params': eps, 'lr': 1e-2}])
train(10000)



# Prediction
Phi_ = Phi(MC_base * 1.j, Ps.to(torch.complex128).to(device))
PhiX = Phi(MC_base * 1.j, X)
A = torch.diag_embed((eps - S_diag).exp()) + PhiX @ PhiX.H
LA = torch.linalg.cholesky(A)
alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)
pred = Phi_.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)
pred = pred.real

pred.detach().cpu().flatten().numpy().tofile('GP.dat')



# Neural network structure, based on PINN paper
class PINN(nn.Module):
    def __init__(self, n_layers = 7, layer_size = 100):
        super(PINN,self).__init__()

        layer_tuple = tuple()
        for i in range(n_layers - 2):
            layer_tuple += (nn.Linear(layer_size, layer_size), nn.Tanh())

        self.layers = nn.Sequential(*(
            (nn.Linear(3, layer_size), nn.Tanh()) +\
            layer_tuple +\
            (nn.Linear(layer_size, 1),))
        )

    def forward(self, x):
        out = self.layers(x)
        return out


# Compute PDE values
def wave(f, x):
    dF, = torch.autograd.grad(f(x).sum(), x, create_graph = True)
    dFxx = torch.autograd.grad(dF[:,0].sum(), x, create_graph = True)[0][:,0]
    dFyy = torch.autograd.grad(dF[:,1].sum(), x, create_graph = True)[0][:,1]
    dFtt = torch.autograd.grad(dF[:,2].sum(), x, create_graph = True)[0][:,2]
    
    PDEs = dFxx + dFyy - dFtt
    return PDEs


# NN parameters
def trainPINN(mask, model, optimizer, loss_func, epochs = 1000, n_collocation = 100, weights = [1.,0.01]):
    Xinit = Ps[mask].to(device)
    Yinit = udata.reshape(-1)[mask].reshape(-1,1).to(device)

    for epoch in range(epochs):
        model.train()
        # Predict on initial data
        pred = model(Xinit)
        loss_data = loss_func(pred, Yinit)

        # Collocation points, unif random in [-2,2]^3 x [0,1]
        coll = torch.rand(n_collocation, 3) * torch.tensor([1,1,1]) + torch.tensor([0,0,0])
        coll = coll.requires_grad_().to(device)
        loss_pde = wave(model,coll).pow(2).sum()

        loss = weights[0] * loss_data + weights[1] * loss_pde

        # if (epoch+1)%1000 == 0:
        #     prediction = model(Ps).detach().squeeze()
        #     sn.heatmap(prediction.reshape(udata.shape), cmap = "vlag", xticklabels=False, yticklabels=False, vmin=-4, vmax=7, cbar = False)
        #     plt.pause(0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        epoch_RMSE = (model(Ps) - udata.reshape(-1,ll)).square().mean().sqrt()
        with torch.no_grad():
            # print
            print(f"Epoch {epoch+1}/{epochs}")
            print(10*"-")
            print(f"Data loss\t{loss_data.detach()}\nPDE Loss:\t{loss_pde.sum().detach()}\nLoss:\t\t{loss.detach()}")
            print(f'PINN RMSE:\t{epoch_RMSE}\n')
            # print(f'Norm:\t{model(Ps).norm()}')

Ps = Ps.to(device)
udata = udata.to(device)
model = PINN(15,200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_func = nn.MSELoss()
trainPINN(mask, model, optimizer, loss_func, 200000, 500, [1000, 1])

model.eval()
pinn_pred = model(Ps).reshape(n_axis,n_axis,n_time).detach()
pinn_pred.cpu().detach().flatten().numpy().tofile('PINN.dat')