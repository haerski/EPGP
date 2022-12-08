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
n_axis = 101
n_time = 51

axis = torch.linspace(-5,5,n_axis, device=device)
time = torch.linspace(0,5,n_time, device=device)
Ps = torch.cartesian_prod(axis,time)
xx, tt = torch.meshgrid((axis, time), indexing="ij")
# Number of points
lP = Ps.shape[0]

# Get random training sets based on parameters.
# This is a mask on the first coordinate of Ps
def randomMask(n_pts = 10, initial = False):
    mask = torch.zeros(lP).bool()
    if not initial:
        mask[torch.randperm(lP)[:n_pts]] = True
    if initial:
        mask[n_time * torch.randperm(n_axis)[:n_pts]] = True
    return mask


# An exact solution
def u(x,t):
    return (np.sqrt(5)*(64*t**3 + 125*(-3 + x)*(-1 + x)*(2 + x) - 50*t*(-2 + x)*(13 \
        + 4*x) + 40*t**2*(16 + 5*x)))/(torch.exp(x**2/(5 + 4*t))*(5 + 4*t)**(7/2))

# Sanity check, our solution solves the PDE
xg = xx.clone().detach().requires_grad_()
tg = tt.clone().detach().requires_grad_()
dx, = autograd.grad(u(xg,tg).sum(), xg, create_graph=True)
dt, = autograd.grad(u(xg,tg).sum(), tg, create_graph=True)
dxx, = autograd.grad(dx.sum(), xg, create_graph=True)
assert((dxx - dt).norm() <= 1e-4)

xx = xx.detach()
tt = tt.detach()
udata = u(xx,tt)
sn.heatmap(udata.cpu(), xticklabels=False, yticklabels=False)
plt.title("True solution")


# Heat EPGP kernel
def k(X,XX):
    x,t = X.split(1,1)
    xx, tt = ( v.T for v in XX.split(1,1) )
    denom = 1+(2*(t+tt))
    return torch.exp(-(x-xx).square() / (2*denom) ) / denom.sqrt()



# Generation
# jitter = 1e-10
# cov = k(Ps,Ps)
# mean = torch.zeros(cov.shape[0])
# d = torch.distributions.MultivariateNormal(mean, cov + jitter * torch.eye(cov.shape[0]))
# sn.heatmap(d.sample().reshape(n_axis,n_time), cbar = False)


def posterior_mean(mask):
    X = Ps[mask]
    Y = u(*X.unbind(1)).view(-1,1)

    kXX = k(X,X)
    k_X = k(Ps,X)

    eps = 1e-6
    A = kXX + eps * torch.eye(kXX.shape[0], device=device)
    L = torch.linalg.cholesky(A)
    alpha = torch.linalg.solve_triangular(L, Y, upper=False)
    alpha1 = torch.linalg.solve_triangular(L.T, alpha, upper=True)

    return k_X @ alpha1


# Posterior
## Initial data
torch.manual_seed(42)
mask = randomMask()
posterior_mean(mask)


### PINN ####

# Neural network structure, based on PINN paper
class PINN(nn.Module):
    def __init__(self, n_layers = 7, layer_size = 100):
        super(PINN,self).__init__()

        layer_tuple = tuple()
        for i in range(n_layers - 2):
            layer_tuple += (nn.Linear(layer_size, layer_size), nn.Tanh())

        self.layers = nn.Sequential(*(
            (nn.Linear(2, layer_size), nn.Tanh()) +\
            layer_tuple +\
            (nn.Linear(layer_size, 1),))
        )

    def forward(self, x):
        out = self.layers(x)
        return out


# Compute PDE values
def heat_eqn(f, x):
    d, = torch.autograd.grad(f(x).sum(), x, create_graph = True)
    dd, = torch.autograd.grad( d[:,0].sum(), x, retain_graph = True)

    PDEs = d[:,1] - dd[:,0]
    return PDEs


# NN parameters
def trainPINN(mask, model, optimizer, loss_func, epochs = 1000, n_collocation = 100, weights = [1.,0.01]):
    Xinit = Ps[mask].to(device)
    Yinit = udata.reshape(-1)[mask].to(device)

    model.train()
    for epoch in range(epochs):
        # Predict on initial data
        pred = model(Xinit)
        loss_data = loss_func(pred.squeeze(), Yinit)

        # Collocation points, unif random in [-5,5] x [0,5]
        coll = torch.rand(n_collocation, 2) * torch.tensor([10, 5]) + torch.tensor([-5,0])
        coll = coll.requires_grad_().to(device)
        loss_pde = heat_eqn(model,coll).pow(2).mean()

        loss = weights[0] * loss_data + weights[1] * loss_pde

        # print
        print(f"Epoch {epoch+1}/{epochs}")
        print(10*"-")
        print(f"Data loss\t{loss_data.detach()}\nPDE Loss:\t{loss_pde.sum().detach()}\nLoss:\t\t{loss.detach()}\n")

        # if (epoch+1)%1000 == 0:
        #     prediction = model(Ps).detach().squeeze()
        #     sn.heatmap(prediction.reshape(udata.shape), cmap = "vlag", xticklabels=False, yticklabels=False, vmin=-4, vmax=7, cbar = False)
        #     plt.pause(0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

model = PINN(7,100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_func = nn.MSELoss()
trainPINN(mask, model, optimizer, loss_func, 10000, 100, [1, 1])

plot_diff(mask, model(Ps).detach().squeeze(), "PINN", with_points=True)



### Comparison
## Few random points
mask = randomMask(16)
print(f"Mask size: {mask.sum().item()}")
# GP method
posterior = posterior_mean(mask).view_as(udata)
# PINN
model = PINN(7,100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_func = nn.MSELoss()
trainPINN(mask, model, optimizer, loss_func, 10000, 500, weights = [1, 0.1])
model.eval()
pinn = model(Ps).detach().view_as(udata)

plt.plot((pinn-udata).square().mean(0).sqrt())
plt.plot((posterior-udata).square().mean(0).sqrt())



#### Data gathering ####
results = pd.DataFrame(columns = ['Model', '# points', 'init', 'pred'])
repeats = 10
i = 0
for rep in range(repeats):
    # GP covar
    # With initial conditions
    init = True
    for maskSize in [6,12,25,50,100]:
        mask = randomMask(maskSize, initial = init)
        print(f"Mask size: {mask.sum().item()}")
        # GP
        posterior = posterior_mean(mask).flatten()
        results.loc[len(results)] = ['EPGP', maskSize, init, posterior.cpu()]
        # PINN
        model = PINN(7,100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        loss_func = nn.MSELoss()
        trainPINN(mask, model, optimizer, loss_func, epochs = 10000, weights = [1, 1])
        results.loc[len(results)] = ['PINN', maskSize, init, model(Ps).detach().flatten().cpu()]
        results.to_pickle("heat.pkl")

    # With random points
    init = False
    for maskSize in [5,10,50,100,500,1000,2500]:
        mask = randomMask(maskSize, initial = init)
        print(f"Mask size: {mask.sum().item()}")
        # GP
        posterior = posterior_mean(mask).flatten()
        results.loc[len(results)] = ['EPGP', maskSize, init, posterior.cpu()]
        # PINN
        model = PINN(7,100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        loss_func = nn.MSELoss()
        trainPINN(mask, model, optimizer, loss_func, epochs = 10000, weights = [1, 1])
        results.loc[len(results)] = ['PINN', maskSize, init, model(Ps).detach().flatten().cpu()]
        results.to_pickle("heat.pkl")


## One instance
torch.manual_seed(13)
mask = randomMask(16)
print(f"Mask size: {mask.sum().item()}")
# GP method
lM = 10000
Ms = getMCPoints(lM)
covar = MCintegration(Ps, Ms, batch_size=10, verbose=True)
covar = covar.real
prior = getPrior(0, covar)
posterior = GPPosterior(mask, prior, covar, verbose=True, with_covar=False, eps=1e-4)
# PINN
model = PINN(7,100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_func = nn.MSELoss()
trainPINN(mask, model, optimizer, loss_func, epochs = 10000, weights = [1, 1])
# Plots
fig, axs = plot_diff(
                mask,
                (posterior.mean, model(Ps).detach().squeeze()),
                ("Gaussian Process", "PINN"),
                with_points=True,
                vmin = -0.1, vmax = 0.1)
plt.savefig("instance.png", dpi = 300)
pd.DataFrame({
    "method": ["GP", "PINN"],
    "pred": [posterior.mean.cpu(), model(Ps).detach().squeeze().cpu()],
    "mask": [mask, mask],
    "pts": [Ps[mask].cpu(), Ps[mask].cpu()]}).to_pickle("instance.pkl")
