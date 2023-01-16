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
import pde
import time as t
from copy import deepcopy

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# torch.set_default_dtype(torch.float32)
torch.set_default_dtype(torch.float64)
torch.manual_seed(13)

grid = pde.grids.CartesianGrid([(-1,1),(-1,1)], [64,64])
XX, YY = np.meshgrid(*grid.axes_coords)

# Initial state for wave equation
ZZ = np.exp(-10*((XX-0.35)**2 + (YY-0.25)**2))
state = pde.ScalarField(grid, ZZ)

# We solve it
eq = pde.pdes.wave.WavePDE()
init = eq.get_initial_condition(state)
storage = pde.MemoryStorage()
# result = eq.solve(init, t_range=3, dt=0.00001, tracker=['consistency','progress',pde.LivePlotTracker(0.1),storage.tracker(0.1)])
result = eq.solve(init, t_range=3, dt=0.00001, tracker=['consistency','progress',storage.tracker(0.1)])

# keep only values, not velocities
u_true = torch.stack([torch.tensor(t[0]) for t in storage.data], -1).to(device)

# convert py-pde types to torch
y_axis = torch.tensor(grid.axes_coords[0])
x_axis = torch.tensor(grid.axes_coords[1])
t_axis = torch.tensor(storage.times)

Ps = torch.cartesian_prod(x_axis,y_axis,t_axis).to(torch.complex128).to(device)

u_true.detach().cpu().numpy().tofile("u_true.dat")
y_axis.detach().cpu().numpy().tofile("y_axis.dat")
x_axis.detach().cpu().numpy().tofile("x_axis.dat")
t_axis.detach().cpu().numpy().tofile("t_axis.dat")



def randomData(n_data):
    randidx = torch.randperm(Ps.shape[0])[:n_data]
    X = Ps[randidx].to(torch.complex128)
    Y = u_true.flatten()[randidx].to(torch.complex128).reshape(-1,1)
    return X,Y


#### S-EPGP ####
def getVarietyPoints(z):
    xx, yy = z.unbind(1)
    tt = (xx.square() + yy.square()).sqrt()
    return torch.stack([torch.stack([xx,yy,tt],1), torch.stack([xx,yy,-tt],1)])

def Phi(Z, X):
    return Z.inner(X).exp().mean(0)

def NLML(X,Y,z,Sigma,eps):
    Z = getVarietyPoints(z).to(torch.complex128)
    PhiX = Phi(Z, X)
    A = torch.diag_embed((eps - Sigma).exp()) + PhiX @ PhiX.H
    LA = torch.linalg.cholesky(A)
    alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)

    nlml = 1/(2*eps.exp()) * (Y.norm().square() - alpha.norm().square())
    nlml += (PhiX.shape[1] - PhiX.shape[0])/2 * eps
    nlml += LA.diag().real.log().sum()
    nlml += 0.5*Sigma.sum()

    return nlml


def trainSEPGP(X,Y,z,Sigma,eps,opt,sched,epoch_max = 1000):
    for epoch in range(epoch_max):
        nlml = NLML(X,Y,z * 1.j, Sigma, eps)
        print(f"{epoch+1}/{epoch_max}\tnlml {nlml.detach():.3f}\tlr {sched.get_last_lr()[0]:.3e}\ts0 {eps.exp().item():.3e}\tz std {z.std().item():.3f}")

        opt.zero_grad()
        nlml.backward()
        opt.step()
        sched.step()

def trainLEPGP(X,Y,z,Sigma,eps,l,opt,sched,epoch_max = 1000):
    for epoch in range(epoch_max):
        nlml = NLML(X,Y,l * z * 1.j, Sigma, eps)
        print(f"{epoch+1}/{epoch_max}\tnlml {nlml.detach():.3f}\tlr {sched.get_last_lr()[0]:.3e}\ts0 {eps.exp().item():.3e}\tl {l.item():.3f}")

        opt.zero_grad()
        nlml.backward()
        opt.step()
        sched.step()


def predictSEPGP(X_,X,Y,z,Sigma,eps):
    with torch.no_grad():
        Z = getVarietyPoints(z * 1.j).to(torch.complex128)
        PhiX = Phi(Z, X)
        A = torch.diag_embed((eps - Sigma).exp()) + PhiX @ PhiX.H
        LA = torch.linalg.cholesky(A)
        alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)
        alpha1 = torch.linalg.solve_triangular(LA.H, alpha, upper = True)

        Phi_ = Phi(Z, X_)
        return (Phi_.H @ alpha1).real


### PINN ###
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
def trainPINN(X, Y, model, optimizer, loss_func, epochs = 1000, n_collocation = 100, weights = [1.,1]):
    for epoch in range(epochs):
        model.train()
        # Predict on initial data
        pred = model(X)
        loss_data = loss_func(pred, Y)

        # Collocation points, unif random in [-1,1]^2 x [0,3]
        coll = torch.rand(n_collocation, 3) * torch.tensor([2,2,3]) + torch.tensor([-1,-1,0])
        coll = coll.requires_grad_().to(device)
        loss_pde = wave(model,coll).pow(2).sum()

        loss = weights[0] * loss_data + weights[1] * loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            print(f"Epoch {epoch+1}/{epochs}")
            print(10*"-")
            print(f"Data loss\t{loss_data.detach()}\nPDE Loss:\t{loss_pde.sum().detach()}\nLoss:\t\t{loss.detach()}")


## Computation templates ##
## imaginary S-EPGP ##
def im_SEPGP(r):
    #r = number of dirac deltas
    z = (2 * torch.randn((r,2), device=device)).requires_grad_()
    Sigma = torch.full((r,), -np.log(r), device=device).requires_grad_()
    eps = torch.tensor(np.log(1e-4), device=device).requires_grad_()

    opt = torch.optim.Adam([
        {'params': z, 'lr': 1e-1},
        {'params': [Sigma,eps], 'lr': 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500)
    trainSEPGP(X,Y,z,Sigma,eps,opt,sched,3000)
    return predictSEPGP(Ps, X, Y, z, Sigma, eps)

## complex S-EPGP ##
def cplx_SEPGP(r):
    # r = number of dirac deltas
    z = (2 * torch.randn((r,2), device=device)).to(torch.complex128).requires_grad_()
    Sigma = torch.full((r,), -np.log(r), device=device).requires_grad_()
    eps = torch.tensor(np.log(1e-4), device=device).requires_grad_()

    opt = torch.optim.Adam([
        {'params': z, 'lr': 1e-1},
        {'params': [Sigma,eps], 'lr': 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500)
    trainSEPGP(X,Y,z,Sigma,eps,opt,sched,3000)
    return predictSEPGP(Ps, X, Y, z, Sigma, eps)
    

## vanilla EPGP ##
def vanilla_EPGP(r):
    # r = number of MC points
    z = (2 * torch.randn((r,2), device=device))
    Sigma = torch.full((r,), -np.log(r), device=device)
    eps = torch.tensor(np.log(1e-4), device=device).requires_grad_()

    opt = torch.optim.Adam([eps], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500)
    trainSEPGP(X,Y,z,Sigma,eps,opt,sched,3000)
    return predictSEPGP(Ps, X, Y, z, Sigma, eps)
    
## EPGP with Length Scale ##
def LS_EPGP(r):
    # r = number of MC points
    z = torch.randn((r,2), device=device)
    l = torch.tensor(np.log(2)).requires_grad_()
    Sigma = torch.full((r,), -np.log(r), device=device)
    eps = torch.tensor(np.log(1e-4), device=device).requires_grad_()

    opt = torch.optim.Adam([
        {'params': l, 'lr': 1e-2},
        {'params': eps, 'lr': 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500)
    trainLEPGP(X,Y,z,Sigma,eps,l,opt,sched,3000)
    return predictSEPGP(Ps, X, Y, l*z, Sigma, eps)
    
## PINN ##
def run_PINN(hid,width):
    model = PINN(hid,width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_func = nn.MSELoss()
    trainPINN(X.real, Y.real, model, optimizer, loss_func, 3000, 500, [1000., 1])

    model.eval()
    return model(Ps.real)



## Runs ##
res = pd.DataFrame(columns=['Model', 'pred', 'Time', 'Training points'])
res.to_pickle("2dwave_comp.pkl")

torch.manual_seed(13)
for repeat in range(10):
    for n_data in [32, 128, 512, 2048]:
        X, Y = randomData(n_data)
        # S-EPGPs
        for r in [32,64,128]:
            # im
            start = t.time()
            pred = im_SEPGP(r)
            end = t.time()
            name = f"S-EPGP ({r}), im"
            res.loc[len(res)] = [name, pred.flatten().cpu(), end-start, n_data]
            res.to_pickle("2dwave_comp.pkl")
            # cplx
            start = t.time()
            pred = cplx_SEPGP(r)
            end = t.time()
            name = f"S-EPGP ({r}), cplx"
            res.loc[len(res)] = [name, pred.flatten().cpu(), end-start, n_data]
            res.to_pickle("2dwave_comp.pkl")

        # EPGPs
        for r in [100, 1000]:
            # vanilla
            start = t.time()
            pred = vanilla_EPGP(r)
            end = t.time()
            name = f"EPGP ({r}), vanilla"
            res.loc[len(res)] = [name, pred.flatten().cpu(), end-start, n_data]
            res.to_pickle("2dwave_comp.pkl")
            # LS
            start = t.time()
            pred = LS_EPGP(r)
            end = t.time()
            name = f"EPGP ({r}), LS"
            res.loc[len(res)] = [name, pred.flatten().cpu(), end-start, n_data]
            res.to_pickle("2dwave_comp.pkl")

        # PINN
        for hid, width in [(15,200), (7,100)]:
            start = t.time()
            pred = run_PINN(hid,width)
            end = t.time()
            name = f"PINN ({hid},{width})"
            res.loc[len(res)] = [name, pred.flatten().cpu().detach(), end-start, n_data]
            res.to_pickle("2dwave_comp.pkl")


## Make table
res = pd.read_pickle("2dwave_comp.pkl_backup")

foo = res
bar = torch.tensor([32,128,512,2048])

res = pd.concat([foo, pd.Series(bar[None,...,None].expand(10,4,12).flatten(), name="Training points")], 1)

res['RMS error'] = res['pred'].map(lambda x: (x.to(device) - u_true.flatten()).square().mean().sqrt().item())
res_clean = res.drop(columns=['pred'])
res_clean = res_clean.groupby(['Model','Training points']).agg(['mean', 'std'])

# RMS table
res_str = res_clean['RMS error'].apply(lambda x: f'${x["mean"]:.3f} \\pm {x["std"]:.3f}$', 1)
res_str = res_str.reset_index().pivot(index = 'Model', columns = 'Training points')
print(res_str.to_latex())

# Time table
time_str = res_clean['Time'].apply(lambda x: f'${x["mean"]:.1f} \\pm {x["std"]:.1f}$', 1)
time_str = time_str.reset_index().pivot(index = 'Model', columns = 'Training points')
print(time_str.to_latex())