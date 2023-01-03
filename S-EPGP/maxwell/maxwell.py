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



# Sampling using Mathematica
axis = torch.tensor(h5py.File('axis.h5')['Dataset1'][...], dtype=torch.float64)
time = torch.tensor(h5py.File('ts.h5')['Dataset1'][...], dtype=torch.float64)

n_axis = axis.shape[0]
n_time = time.shape[0]

Ps = torch.cartesian_prod(axis,axis,axis,time).to(device)
# Number of points
lP = Ps.shape[0]
# Output dimension
ll = 6

# Get random training sets based on parameters.
# This is a flattened version of EBdata
# full = True will give result in both E and B for each time point
def randomMask(n_pts = 10, initial = False, full = True):
    mask = torch.zeros(lP*ll).bool()
    if full and not initial:
        idx = (ll * torch.randperm(lP)[:n_pts].unsqueeze(1)) + torch.arange(0,ll)
        mask[idx.reshape(-1)] = True

    if not full and not initial:
        idx = torch.randperm(lP*ll)[:n_pts]
        mask[idx] = True

    if full and initial:
        idx = ll * n_time * torch.randperm(n_axis**3)[:n_pts].unsqueeze(1) + torch.arange(0,ll)
        mask[idx.reshape(-1)] = True

    if not full and initial:
        idx = n_time * torch.randperm(ll * n_axis**3)[:n_pts]
        mask[idx] = True
        mask = mask.reshape(n_axis, n_axis, n_axis, ll, n_time).permute([0,1,2,4,3]).reshape(-1)    
    return mask

# An exact solution
Edata = torch.tensor(h5py.File('EEdata.h5')['Dataset1'][...], dtype=torch.float64)
Bdata = torch.tensor(h5py.File('BBdata.h5')['Dataset1'][...], dtype=torch.float64)
EBdata = torch.cat([Edata, Bdata], 4)
EBdata = EBdata.to(device)
EBdata.shape

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




### S-EPGP ###
class SEPGP:
    def __init__(self,n_spectral, n_ops = 6, lr = 1e-2):
        self.n_spectral = n_spectral
        self.n_ops = n_ops
        self.base_pts = torch.randn((self.n_ops, self.n_spectral, 3), device=device).requires_grad_()
        self.eps = torch.tensor(np.log(1e-6), device=device).requires_grad_()
        self.S_diag = torch.full((self.n_spectral*self.n_ops,), 0., device=device).requires_grad_()

        self.opt = torch.optim.Adam([self.base_pts, self.eps, self.S_diag], lr = lr)

    def train(self,X,Y,epochs = 10000):
        for epoch in range(epochs):
            Z = getVarPoints(self.base_pts.flatten(0,1)).unflatten(1, (self.n_ops, self.n_spectral))
            PhiX = Phi(Z * 1.j, X)
            A = (self.n_ops * self.n_spectral) * torch.diag_embed((self.eps - self.S_diag).exp()) + PhiX @ PhiX.H
            LA = torch.linalg.cholesky(A)
            alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)

            nlml = 1/(2*self.eps.exp()) * (Y.norm().square() - alpha.norm().square())
            nlml += (PhiX.shape[1] - PhiX.shape[0])/2 * self.eps
            nlml += LA.diag().real.log().sum()
            nlml += 0.5*self.S_diag.sum()

            self.opt.zero_grad()
            nlml.backward()
            self.opt.step()

            with torch.no_grad():
                train_pred = PhiX.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)
                err = (train_pred.real - Y).square().mean().sqrt()
                print(26*"~" + f'\nepoch {epoch}\n\
nlml {nlml}\n\
err {err}\n\
eps {self.eps.exp()}\n')

    def predict(self, X, Y):
        Z = getVarPoints(self.base_pts.flatten(0,1)).unflatten(1, (self.n_ops, self.n_spectral))
        Phi_ = Phi(Z * 1.j, Ps).to(device)
        PhiX = Phi(Z * 1.j, X)
        A = (self.n_ops * self.n_spectral) * torch.diag_embed((self.eps - self.S_diag).exp()) + PhiX @ PhiX.H
        LA = torch.linalg.cholesky(A)
        alpha = torch.linalg.solve_triangular(LA, PhiX @ Y.to(torch.complex128), upper=False)
        pred = Phi_.H @ torch.linalg.solve_triangular(LA.H, alpha, upper=True)
        return pred.real.view_as(EBdata)

def make_initial_data(n):
    mask = randomMask(n)
    X = Ps[mask.reshape(-1,ll).all(1)]
    Y = EBdata.flatten()[mask].reshape(-1,1)
    return X,Y


# Initial data
torch.manual_seed(42)
X,Y = make_initial_data(1000)
model = SEPGP(24, lr=1e-2)
model.train(X,Y,10000)
pred = model.predict(X,Y)
(pred-EBdata).square().mean().sqrt()



pred.detach().cpu().numpy().tofile("pred.dat")
axis.cpu().numpy().tofile("axis.dat")
time.cpu().numpy().tofile("time.dat")








### PINN ####

# Neural network structure, based on PINN paper
class PINN(nn.Module):
    def __init__(self, n_layers = 7, layer_size = 100):
        super(PINN,self).__init__()

        layer_tuple = tuple()
        for i in range(n_layers - 2):
            layer_tuple += (nn.Linear(layer_size, layer_size), nn.Tanh())

        self.layers = nn.Sequential(*(
            (nn.Linear(4, layer_size), nn.Tanh()) +\
            layer_tuple +\
            (nn.Linear(layer_size, 6),))
        )

    def forward(self, x):
        out = self.layers(x)
        return out


# Compute PDE values
def maxwell(f, x):
    dEx, = torch.autograd.grad(f(x)[:,0].sum(), x, create_graph = True)
    dEy, = torch.autograd.grad(f(x)[:,1].sum(), x, create_graph = True)
    dEz, = torch.autograd.grad(f(x)[:,2].sum(), x, create_graph = True)
    dBx, = torch.autograd.grad(f(x)[:,3].sum(), x, create_graph = True)
    dBy, = torch.autograd.grad(f(x)[:,4].sum(), x, create_graph = True)
    dBz, = torch.autograd.grad(f(x)[:,5].sum(), x, create_graph = True)

    PDEs = torch.stack([
        dEx[:,0] + dEy[:,1] + dEz[:,2], # div E = 0
        -dEy[:,2] + dEz[:,1] + dBx[:,3],
        dEx[:,2] - dEz[:,0] + dBy[:,3],
        -dEx[:,1] + dEy[:,0] + dBz[:,3], # curl E = - dB/dt
        dBx[:,0] + dBy[:,1] + dBz[:,2], # div B = 0
        -dBy[:,2] + dBz[:,1] - dEx[:,3],
        dBx[:,2] - dBz[:,0] - dEy[:,3],
        -dBx[:,1] + dBy[:,0] - dEz[:,3] # curl B = dE/dt
        ])
    return PDEs


# NN parameters
def trainPINN(model, optimizer, loss_func, epochs = 1000, n_collocation = 100, weights = [1.,0.01]):
    Xinit = X
    Yinit = Y.reshape(-1,6)

    if(Xinit.shape[0] != Yinit.shape[0]):
        raise ValueError("PINN needs to be trained on all outputs simultaneously")

    for epoch in range(epochs):
        model.train()
        # Predict on initial data
        pred = model(Xinit)
        loss_data = loss_func(pred, Yinit)

        # Collocation points, unif random in [-1,1]^3 x [0,2]
        coll = torch.rand(n_collocation, 4) * torch.tensor([2,2,2,2]) + torch.tensor([-1,-1,-1,0])
        coll = coll.requires_grad_().to(device)
        loss_pde = maxwell(model,coll).pow(2).mean(1).sum()

        loss = weights[0] * loss_data + weights[1] * loss_pde

        # if (epoch+1)%1000 == 0:
        #     prediction = model(Ps).detach().squeeze()
        #     sn.heatmap(prediction.reshape(udata.shape), cmap = "vlag", xticklabels=False, yticklabels=False, vmin=-4, vmax=7, cbar = False)
        #     plt.pause(0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        epoch_RMSE = (model(Ps) - EBdata.reshape(-1,ll)).square().mean().sqrt()
        with torch.no_grad():
            # print
            print(f"Epoch {epoch+1}/{epochs}")
            print(10*"-")
            print(f"Data loss\t{loss_data.detach()}\nPDE Loss:\t{loss_pde.sum().detach()}\nLoss:\t\t{loss.detach()}")
            print(f'PINN RMSE:\t{epoch_RMSE}\n')


model = PINN(7,100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_func = nn.MSELoss()
trainPINN(model, optimizer, loss_func, 9000, 500, [1, 1])
optmizer = torch.optim.LBFGS(model.parameters(), lr=1)
trainPINN(model, optimizer, loss_func, 1000, 500, [1, 1])

model.eval()
model(Ps).detach().flatten().numpy().tofile('PINN.dat')


### Data gathering
torch.manual_seed(42)
ep_results = pd.DataFrame(columns = ['n_MC', 'n_data', 'time', 'result'])
pinn_results = pd.DataFrame(columns = ['width', 'n_data', 'time', 'result'])

for repeat in range(10):
    for n_data in [5,10,50,100,1000]:
        X,Y = make_initial_data(n_data)
        for n_mc in [4,8,16,32,64]:
            model = SEPGP(n_mc, lr=1e-2)
            start = t.time()
            model.train(X,Y,10000)
            end = t.time()
            pred = model.predict(X,Y)
            ep_results.loc[len(ep_results)] = [n_mc, n_data, end-start, pred.cpu().detach()]
            ep_results.to_pickle("ep_results.pkl")
        for width in [50,100,200]:
            model = PINN(7,width).to(device)
            loss_func = nn.MSELoss()
            start = t.time()
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
            loss_func = nn.MSELoss()
            trainPINN(model, optimizer, loss_func, 9000, 500, [1, 1])
            optmizer = torch.optim.LBFGS(model.parameters(), lr=1)
            trainPINN(model, optimizer, loss_func, 1000, 500, [1, 1])
            end = t.time()
            model.eval()
            pred = model(Ps).view_as(EBdata)
            pinn_results.loc[len(pinn_results)] = [width, n_data, end-start, pred.cpu().detach()]
            pinn_results.to_pickle("pinn_results.pkl")


            




pinn_results.loc[len(pinn_results)] = [1,2,pred]

X,Y = make_initial_data(100)
model = SEPGP(12)
model.train(X,Y,10000)
pred = model.predict(X,Y)
(pred-EBdata).square().mean().sqrt()