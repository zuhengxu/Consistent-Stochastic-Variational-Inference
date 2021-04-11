from autograd.scipy import stats
import autograd.numpy as np 
from scipy.stats import norm
import os, sys
sys.path.append("/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code")
from VI.util import *




# gaussian mixture logpdf: 0.7N(0, 2) + 0.15N(30,3) + 0.15N(-30,3)
def log_gs_mix(x):
    lw = np.log(np.array([0.7,0.15,0.15]))
    lpdf = np.hstack((stats.norm.logpdf(x, 0, 2)[:, np.newaxis], 
                    stats.norm.logpdf(x, 30, 3)[:, np.newaxis], 
                    stats.norm.logpdf(x, -30, 3)[:, np.newaxis] ))
    return logsumexp(lw + lpdf, axis= 1)

gaussian_mix = lambda x: np.exp(log_gs_mix(x))

# smoothed gaussian mixture log/pdf: with smoothing kernel N(0, 10)
def log_smth_gs_mix(x):
    lw = np.log(np.array([0.7,0.15,0.15]))
    lpdf = np.hstack((stats.norm.logpdf(x, 0, np.sqrt(104))[:, np.newaxis], 
            stats.norm.logpdf(x, 30, np.sqrt(109))[:, np.newaxis], 
            stats.norm.logpdf(x, -30, np.sqrt(109))[:, np.newaxis] ))
    return logsumexp(lw + lpdf, axis= 1)


# log pdf for synthetic bayesian model
def toy_unorm_lpdf(x, sample_size):
    # np.random.seed(2020)
    n = sample_size
    lw = np.log(np.array([0.2,0.2,0.2, 0.2, 0.2]))
    lpdf = np.hstack((stats.norm.logpdf(x, -1, 0.15)[:, np.newaxis], 
            stats.norm.logpdf(x, 1, 0.1)[:, np.newaxis], 
            stats.norm.logpdf(x, -4, 0.3)[:, np.newaxis],
            stats.norm.logpdf(x, 4, 0.3)[:, np.newaxis],
            stats.norm.logpdf(x, -8, 0.1)[:, np.newaxis] ))

    log_prior = logsumexp(lw + lpdf, axis= 1)

    r = np.random.normal(size = n)*np.sqrt(10) + 3
    llh =  -(np.sum(r**2)- 2* np.sum(r)*x + n*(x**2))/10000
    # likelihood = (np.exp(-(x-5)**2/(2*2))/np.sqrt(2*np.pi*2))**n
    #

    return log_prior + llh



