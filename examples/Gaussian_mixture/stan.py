from arviz.utils import _var_names
import autograd.numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import pystan
import arviz
from arviz.utils import _var_names
import matplotlib.pyplot as plt
import os, pickle, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


#####################################
# Load and visualize dataset
#####################################
# read synthetic dataset
syn_dat = np.load(os.path.join(sys.path[1], 'data/syndat_gmm.npy'))
X = syn_dat[:, :2]
label = syn_dat[:, -1]
# # Plot the data with K Means Labels
# p_truedat = plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=label, s=40, cmap='viridis')
# p_truedat.savefig(os.path.join(sys.path[0], 'figures/syn_Dat.png'))


GMM_syndat = {
    'N': 400,
    'K': 3,
    'D': 2,
    'alpha0': 1,
    'mu_sigma0': 1,
    'sigma_sigma0': 1,
    'y': X
}

result_path = os.path.join(sys.path[0], 'results/stan')
fig_path = os.path.join(sys.path[0], 'figures')

#####################################
# GMM without ordering
#####################################

stan_GMM = """
data {
    int<lower = 0> N; //number of obs
    int<lower = 0> K; //number of components
    int<lower = 0> D; //dimension
    vector[D] y[N]; //obs

    real<lower = 0> alpha0; //dir prior param
    real<lower = 0> mu_sigma0; //means prior
    real<lower = 0> sigma_sigma0; // var prior
}

transformed data{
    vector<lower = 0>[K] alpha0_vec;
    for (k in 1:K){
        alpha0_vec[k] = alpha0;
    }
}

parameters{
    vector[K] lambda;
    vector[D] mu[K];
    vector[D] log_sigma[K];
}

transformed parameters{
    vector<lower = 0>[K] exp_lmd;
    simplex[K] theta = softmax(lambda);
    for (k in 1:K){
        exp_lmd[k] = exp(lambda[k]);
    }
}

model{
    //prior
    for (k in 1:K){
        exp_lmd[k] ~ gamma(alpha0, 1);
        mu[k] ~ normal(0.0 , mu_sigma0);
        log_sigma[k] ~ normal(0.0, sigma_sigma0);
        target += log(exp_lmd[k]);
    }

    //likelihood
    for (n in 1:N){
        real ps[K];
        for (k in 1:K){
            ps[k] = log(theta[k]) + normal_lpdf(y[n]| mu[k], exp(log_sigma[k]));
        }
        target += log_sum_exp(ps);
    }
}
"""

# compile model if needed
if not os.path.exists( os.path.join(result_path,'stan_GMM_model.pkl')):
    # Compile the model
    sm = pystan.StanModel(model_code=stan_GMM)
    # Save the model
    with open(os.path.join(result_path,'stan_GMM_model.pkl'), 'wb') as f:
        pickle.dump(sm, f)
else:
    sm = pickle.load(open(os.path.join(result_path,'stan_GMM_model.pkl'), 'rb'))

# fit the model if needed and get the trace samples
if not os.path.exists( os.path.join(result_path, 'GMM_trace.pkl')):
    fit = sm.sampling(data=GMM_syndat, iter=20000, chains=8)
    trace = fit.extract(permuted=True)
    with open(os.path.join(result_path, 'GMM_trace.pkl'), 'wb') as f:
        pickle.dump(trace,f)
else:
    trace = pickle.load(open(os.path.join(result_path, 'GMM_trace.pkl'), 'rb'))


# trace.keys()
# trace['mu'][:,0,0]





# # marginal posterior plots
# axe1 = arviz.plot_posterior(fit)
# fig1 = axe1.ravel()[0].figure
# fig1.savefig(fig_path + "/posts.png")

# # fullrank VB (not really) using ADVI
# gmm_vb = sm.vb(data = GMM_syndat, algorithm= 'fullrank', iter= 100000, eval_elbo = 2000)
# print(gmm_vb['inits'])



















# #####################################
# ### GMM with ordering
# #####################################

# stan_GMM_order = """
# data {
#     int<lower = 0> N; //number of obs
#     int<lower = 0> K; //number of components
#     int<lower = 0> D; //dimension
#     vector[D] y[N]; //obs

#     real<lower = 0> alpha0; //dir prior param
#     real<lower = 0> mu_sigma0; //means prior
#     real<lower = 0> sigma_sigma0; // var prior
# }

# transformed data{
#     vector<lower = 0>[K] alpha0_vec;
#     for (k in 1:K){
#         alpha0_vec[k] = alpha0;
#     }
# }

# parameters{
#     vector[K] lambda;
#     ordered[K] mu_raw[D];
#     vector[D] log_sigma[K];
# }

# transformed parameters{
#     vector[D] mu[K];
#     vector<lower = 0>[K] exp_lmd;
#     simplex[K] theta = softmax(lambda);

#     for (d in 1:D){
#         for (k in 1:K){
#             mu[k,d] = mu_raw[d,k];
#         }
#     }

#     for (k in 1:K){
#         exp_lmd[k] = exp(lambda[k]);
#     }
# }

# model{
#     //prior
#     for (k in 1:K){
#         exp_lmd[k] ~ gamma(alpha0, 1);
#         for (d in 1:D){
#             mu[k, d] ~ normal(0.0 , mu_sigma0);
#         }
#         log_sigma[k] ~ normal(0.0, sigma_sigma0);
#         target += log(exp_lmd[k]);
#     }

#     //likelihood
#     for (n in 1:N){
#         real ps[K];
#         for (k in 1:K){
#             ps[k] = log(theta[k]) + normal_lpdf(y[n]| mu[k], exp(log_sigma[k]));
#         }
#         target += log_sum_exp(ps);
#     }
# }
# """
# stan_gmm_order = pystan.StanModel(model_code= stan_GMM_order)
# GMM_order_mcmc = stan_gmm_order.sampling(data = GMM_syndat, iter = 20000, chains = 8)
# gmm_order_samples= GMM_order_mcmc.extract(permuted  =True)
# print(GMM_order_mcmc)
# #plot the post pdf est for mu
# axes2 = arviz.plot_posterior(GMM_order_mcmc)
# fig2 = axes2.ravel()[0].figure
# fig2.savefig(fig_path + "/order_pdfs.png")

# # # vb
# # gmm_order_vb = stan_gmm_order.vb(data = GMM_syndat, algorithm= 'fullrank', iter= 100000, eval_elbo = 2000)
