import autograd.numpy as np
from VI.util import *




def log_likelihood(param, data, sigma):
    X, Y = data[:,1:],  data[:,0][:, None]
    param = np.atleast_2d(param) #param is listed by row
    S = param.shape[0]
    res = np.tile(Y,(1,S)) - np.dot(X, param.T)
    return -0.5/sigma**2 * np.sum(res**2, axis = 0)



def log_prior(param, tau1, tau2):
    param = np.atleast_2d(param)
    D = param.shape[1]
    S = param.shape[0]
    T = np.array([tau1, tau2])[:,None, None]
    P = param[None,:, :]

    A = np.log1p(0.5/T - 1) - 0.5/(T**2) * P**2
    return np.sum(logsumexp( A, axis = 0), axis =1)

# from scipy.stats import mvnorm 
# def sci_lpdf(param, t1, t2):
#     # d = param.shape[1]
#     rv1 = mvnorm(0, t1)
#     rv2 = mvnorm(0, t2)
#     return rv1.logpdf(param) + rv2.logpdf(param)


# sci_lpdf(x0, 1, 10)

def log_posterior(param,data, sigma, tau1,tau2):
    return log_likelihood(param, data, sigma) + log_prior(param, tau1, tau2)


def prior_sample(Sample_size, D, tau1, tau2):
    beta1 = np.random.randn(Sample_size*D)*tau1
    beta2 = np.random.randn(Sample_size*D)*tau2
    Beta = np.vstack((beta1, beta2))
    ind = np.random.choice(np.array([0,1]),size= Sample_size*D, p = [0.5, 0.5])
    Ind = np.vstack((ind, 1 - ind))
    return np.diag(np.dot(Ind.T, Beta)).reshape(Sample_size, D)


def syn_lpdf(param):
    syn_dat = np.load( '../data/syndat_sparse_reg.npy')
    return log_posterior(param, syn_dat, 5, .1, 10)

def syn_pdf(param):
    lpdf = syn_lpdf(param)
    return np.exp(lpdf)


def real_lpdf(param):
    dat = np.load( '../data/prostate.npy')
    return log_posterior(param, dat, 1, .01, 5)

def real_pdf(param):
    lpdf = real_lpdf(param)/5 + 500 # /5 and +550 for numerical stability of SMAP
    return np.expm1(lpdf) + 1





