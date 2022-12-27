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


axis = torch.tensor(h5py.File('axis.h5')['Dataset1'][...], dtype=torch.float64)
time = torch.tensor(h5py.File('ts.h5')['Dataset1'][...], dtype=torch.float64)

n_axis = axis.shape[0]
n_time = time.shape[0]

Ps = torch.cartesian_prod(axis,axis,axis,time).to(device)
# Number of points
lP = Ps.shape[0]
# Output dimension
ll = 6

Edata = torch.tensor(h5py.File('EEdata.h5')['Dataset1'][...], dtype=torch.float64)
Bdata = torch.tensor(h5py.File('BBdata.h5')['Dataset1'][...], dtype=torch.float64)
EBdata = torch.cat([Edata, Bdata], 4)
EBdata = EBdata.to(device)



ep = pd.read_pickle("ep_results.pkl")
ep['rms'] = ep['result'].map(lambda x: (x-EBdata).square().mean().sqrt())
ep_clean = ep.drop(columns=['result'])
ep_agg = ep_clean.groupby(['n_MC','n_data']).agg(['mean','std'])




pinn = pd.read_pickle("pinn_results.pkl")
pinn['rms'] = pinn['result'].map(lambda x: (x-EBdata).square().mean().sqrt())
pinn_clean = pinn.drop(columns=['result'])
pinn_clean.groupby(['width','n_data']).agg(['mean','std'])

pinn_agg = pinn_clean.groupby(['width','n_data']).agg(['mean','std'])
pinn_agg['rms'].unstack('width').swaplevel(0,1,axis=1).T.sort_index()


print(ep_agg['rms'].\
    apply(lambda x: f"{x['mean']:.3g} \\pm {x['std']:.3g}", 1).\
    unstack('n_data').\
    to_latex())

print(pinn_agg['rms'].\
    apply(lambda x: f"{x['mean']:.3g} \\pm {x['std']:.3g}", 1).\
    unstack('n_data').\
    to_latex())
