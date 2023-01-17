# EPGP
Ehrenpreis-Palamodov Gaussian Processes for solving PDE. 

- arXiv link: https://arxiv.org/abs/2212.14319
- Animated demos: https://mathrepo.mis.mpg.de/EPGP/index.html

This repository contains example implementations of EPGP and S-EPGP.
We provide code to generate figures and tables in the [paper](https://arxiv.org/abs/2212.14319), as well as code for generating the animations hosted in the [MathRepo](https://mathrepo.mis.mpg.de/EPGP/index.html).
We also provide a self-contained implementation in the Jupyter notebook [demo.ipynb](demo.ipynb).

### Directory listing

```
.
├── EPGP
│   ├── 1Dheat              # 1D heat equation graphs, with PINN comparison
│   ├── 2Dheat              # 2D heat equation, with varying Gaussian measures on EPGP
│   ├── 2Dwave_comparison   # Table comparing EPGP, S-EPGP, and PINN on 2D waves
│   └── maxwell_generate    # Generating a solution of Maxwell's equations
└── S-EPGP
    ├── 2Dwave_learn        # Learning a vibrating membrane from data
    ├── 2Dwave_solve        # Solving a 2D wave from initial data
    └── maxwell             # Comparison between S-EPGP and PINN for Maxwell's equations
```
