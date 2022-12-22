# Consistent-Stochastic-Variational-Inference

This repository provides source code for the experiments in 

Z. Xu and T. Campbell, [The computatioal asymptotics of Gaussian variational inference](https://link.springer.com/article/10.1007/s11222-022-10125-y). 

Examples run and generate output using Python3.
- `VI/` provides functions for inferences (CSVI/SVI and smoothed MAP) 
- `examples/` provides code to replicate examples and figures
- Datasets(both raw and processed) used in experiments are provided in `examples/data/`, including code that generates the synthetic datasets and processes real datasets

## How to run the code
Each experiment should be run in its own folder (`examples/synthetic_mixture/`, `examples/sparse_regression/`, and `examples/Gaussian_mixture/`): 
- first run `./run.sh` to perform the experiment
- then run `python3 plot.py` to generate plots

**Note:** In `examples/sparse_regression/`, run `python3 stan.py` before running `python3 plot.py`---one of the plots uses samples from [PyStan](https://pystan.readthedocs.io/en/latest/).  


