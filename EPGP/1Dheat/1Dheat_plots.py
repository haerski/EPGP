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
torch.set_default_dtype(torch.float64)


# # Plot options
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "Helvetica"
# })

# Sampling parameters etc
n_axis = 101
n_time = 51

axis = torch.linspace(-5,5,n_axis, device=device)
time = torch.linspace(0,5,n_time, device=device)
Ps = torch.cartesian_prod(axis,time)
xx, tt = torch.meshgrid((axis, time), indexing="ij")
# Number of points
lP = Ps.shape[0]

# An exact solution
def u(x,t):
    return (np.sqrt(5)*(64*t**3 + 125*(-3 + x)*(-1 + x)*(2 + x) - 50*t*(-2 + x)*(13 \
        + 4*x) + 40*t**2*(16 + 5*x)))/(torch.exp(x**2/(5 + 4*t))*(5 + 4*t)**(7/2))

udata = u(xx,tt)
sn.heatmap(udata.cpu(), xticklabels=False, yticklabels=False)
plt.title("True solution")



#### RMS error by time ####
results = pd.read_pickle("heat.pkl")

def RMS_by_time(x):
    return (x.view_as(udata) - udata).square().mean(0).sqrt().numpy()

results["RMS error"] = results["pred"].map(RMS_by_time)
results.drop("pred", 1, inplace=True)
results["t"] = [time.numpy()] * len(results)

results = results.explode(["RMS error","t"]).reset_index()
results["Model"] = results["Model"].map({"EPGP": "EPGP (ours)", "PINN": "PINN"})

### Plotting
cur = results[results["init"]==False]
# cur = results[results["init"]==False][:100]
palette = sn.cubehelix_palette(light=.8, dark=0, rot=-.5, n_colors= (cur["# points"].unique().size) )
g = sn.relplot(
    data = cur,
    kind = "line",
    x = "t",
    y = "RMS error",
    hue = "# points",
    col = "Model",
    palette = palette,
    height=2.5
)
g.set(yscale="log")
g.fig.suptitle('Random points at t in [0,5]')
g.fig.subplots_adjust(top=0.80)
g.savefig("init_false.pdf")


cur = results[results["init"]==True]
palette = sn.cubehelix_palette(light=.8, dark=0, rot=-.5, n_colors= (cur["# points"].unique().size) )
g = sn.relplot(
    data = cur,
    kind = "line",
    x = "t",
    y = "RMS error",
    hue = "# points",
    col = "Model",
    palette = palette,
    height=2.5
)
g.set(yscale="log")
g.fig.suptitle('Random points at t = 0')
g.fig.subplots_adjust(top=0.80)
g.savefig("init_true.pdf")



#### RMS error by model ####
results = pd.read_pickle("heat.pkl")

def RMS_by_model(x):
    return (x.view_as(udata) - udata).square().mean().sqrt().item()

results["RMS error"] = results["pred"].map(RMS_by_model)
results.drop("pred", 1, inplace=True)
results["Model"] = results["Model"].map({"EPGP": "EPGP (ours)", "PINN": "PINN"})

### Plotting
cur = results
g = sn.relplot(
    data = cur,
    kind = "line",
    x = "# points",
    y = "RMS error",
    col = "init",
    style = "Model",
    hue = "Model",
    palette = "dark",
    markers = True,
    facet_kws = {"sharex": False},
    height = 4 )
g.set(yscale="log")
g.savefig("RMS_vs_pts.pdf")