import autograd.numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
from pystan import StanModel
import matplotlib.pyplot as plt
import arviz as az
import os, sys, pickle, pystan
sys.path.insert(1, os.path.join(sys.path[0], '..'))
np.random.seed(2021)



# Folders for resutls/fig
result_path = 'results/stan/'
fig_path = 'figures/'

if not os.path.exists(result_path):
    os.mkdir(result_path)

if not os.path.exists(fig_path):
      os.mkdir(fig_path)



###########################
### load dataset
###########################

syn_dat = np.load(os.path.join(sys.path[1], 'data/syndat_sparse_reg.npy'))
# syn data dict to fit
sparse_syn_dat = {
    'N': syn_dat.shape[0],
    'd': syn_dat.shape[1]-1,
    'X': syn_dat[:,1:],
    'y': syn_dat[:,0],
    'sigma': 5,
    'tau1': 0.1,
    'tau2': 10
}



real_dat = np.load(os.path.join(sys.path[1], 'data/prostate.npy'))
# real data dict to fit
sparse_real_dat = {
    'N': real_dat.shape[0],
    'd': real_dat.shape[1]-1,
    'X': real_dat[:,1:],
    'y': real_dat[:,0],
    'sigma': 1,
    'tau1': 0.01,
    'tau2': 5
}

###########################
### stan model
###########################
spike_slab = """
data {
  int<lower=1> N; // Number of data
  int<lower=1> d; // Number of covariates
  matrix[N,d] X;
  real y[N];
  real<lower = 0> sigma; // sd of Y
  real<lower = 0> tau1; // sd for narrow normal prior
  real<lower = 0 > tau2; // sd for fat normal prior
}

parameters{
    vector[d] beta; //coeff vector
}

model{
    //prior for beta
    matrix[d,2] ps;
    for (i in 1:d){
      ps[i,1] = normal_lpdf(beta[i]| 0, tau1)+ log(0.5);
      ps[i,2] = normal_lpdf(beta[i]|0, tau2)+ log(0.5);
      target += log_sum_exp(ps[i,]);
    }


    //likelihood
    y ~ normal(X * beta, sigma) ;
}
"""

# compile model if needed
if not os.path.exists( os.path.join(result_path,'stan_sr_model.pkl')):
    # Compile the model
    sm = pystan.StanModel(model_code=spike_slab)
    # Save the model
    with open(os.path.join(result_path,'stan_sr_model.pkl'), 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open(os.path.join(result_path,'stan_sr_model.pkl'), 'rb'))



# fit the syn model if needed and get the trace samples
if not os.path.exists( os.path.join(result_path, 'sr_trace.pkl')):
    fit = sm.sampling(data=sparse_syn_dat,iter = 80000, chains = 8,
                      control={'adapt_delta': 0.99, 'max_treedepth': 15})
    trace = fit.extract(permuted=True)
    with open(os.path.join(result_path, 'sr_trace.pkl'), 'wb') as f:
        pickle.dump(trace,f)
    #samples
    samples  = az.from_pystan(fit)
    samples.to_netcdf(result_path + 'syn_trace.nc')
    #posterior marginal kde plot
    samples_read= az.from_netcdf(result_path + 'syn_trace.nc')
    az.rcParams["plot.max_subplots"] = 20
    axe = az.plot_posterior(samples_read, var_names=('beta'), bw = 0.1)
    fig = axe.ravel()[0].figure
    fig.savefig(result_path + "beta_syn.png")
else:
    trace = pickle.load(open(os.path.join(result_path, 'sr_trace.pkl'), 'rb'))

# fit the real model if needed and get the trace samples
if not os.path.exists( os.path.join(result_path, 'sr_real_trace.pkl')):
    fit = sm.sampling(data=sparse_real_dat,iter = 80000, chains = 8,
                      control={'adapt_delta': 0.99, 'max_treedepth': 15})
    trace = fit.extract(permuted=True)
    with open(os.path.join(result_path, 'sr_real_trace.pkl'), 'wb') as f:
        pickle.dump(trace,f)
    
    #samples
    samples  = az.from_pystan(fit)
    samples.to_netcdf(result_path + 'real_trace.nc')
    #posterior marginal kde plot
    samples_read= az.from_netcdf(result_path + 'real_trace.nc')
    az.rcParams["plot.max_subplots"] = 20
    axe = az.plot_posterior(samples_read, var_names=('beta'), bw = 0.1)
    fig = axe.ravel()[0].figure
    fig.savefig(result_path + "beta_real.png")
else:
    trace = pickle.load(open(os.path.join(result_path, 'sr_real_trace.pkl'), 'rb'))









# #samples
# samples  = az.from_pystan(fit)
# samples.to_netcdf(result_path + 'samples.nc')

# #plot
# samples_read= az.from_netcdf(result_path + 'samples.nc')
# az.summary(samples_read.posterior.beta).to_csv(result_path + 'beta_summary.csv')
# az.rcParams["plot.max_subplots"] = 20
# axe = az.plot_posterior(samples_read, var_names=('beta'), bw = 0.1)
# fig = axe.ravel()[0].figure
# fig.savefig(fig_path + "beta.png")




