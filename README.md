# Consistent-Stochastic-Variational-Inference

This repository provides source code for the experiments in our paper Z.Xu and T.Campbell, "[The
computatioal asymptotics of Gaussian variational
inference](https://arxiv.org/abs/2104.05886)". Examples run and generate output
using Python3.
- `VI/` provides functions for inferences (CSVI/SVI and smoothed MAP) 
- `examples/` provides code to replicate examples and figures for [the paper](https://arxiv.org/abs/2104.05886)
- Datasets(both raw and processed) used in experiments are provided in `examples/data/`, including code that generates the synthetic datasets and processes real datasets




## How to run the code?
All experiments should be run in its own folder (`examples/synthetic_mixutre/` `examples/sparse_regression/` and `examples/Gaussian_mixture/`): 
- In each `examples/*` folder, run `./run.sh` to perform the experiment
- In each `examples/*` folder, run `python3 plot.py` to generate plots

**Note:** In `examples/sparse_regression/`, run `python3 stan.py` before running `python3 plot.py`---one of the plot uses Monte Carlo samples generated using [PyStan](https://pystan.readthedocs.io/en/latest/).  


