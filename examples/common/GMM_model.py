# from sklearn.datasets.samples_generator import make_blobs
from VI.util import *
import autograd.numpy as np


def gen_synthetic(N, k, cluster_std, **kwargs):
    if 'random_state' in kwargs:
        r = kwargs.pop('random_state')
    else:
        r = 0
    X, y_true = make_blobs(n_samples=N, centers=k,
                           cluster_std=cluster_std, random_state=r)
    X = X[:, ::-1]  # flip axes for better plotting
    return X, y_true


def log_prior(param, X, K, alpha=1):
    param = np.atleast_2d(param)
    N = X.shape[0]
    D = X.shape[1]
    Mu = param[:, :K*D]
    Lam = param[:, K*D:K*D + K]
    Tau = param[:, -K*D:]

    log_prior = -0.5*np.sum(Mu**2, axis=1) + np.sum(alpha * Lam - np.expm1(Lam) - 1., axis=1) - 0.5*np.sum(Tau**2, axis=1)
    return log_prior



def log_llh(param, X, K):
    param = np.atleast_2d(param)
    mcS = param.shape[0]
    N = X.shape[0]
    D = X.shape[1]
    Mu = param[:, :K*D]
    Lam = param[:, K*D:K*D + K]
    Tau = param[:, -K*D:]

    a = Lam - logsumexp(Lam, axis=1)[:, None]
    b = -0.5*(np.sum(Tau.reshape(mcS, K, D)**2, axis=2))
    t1 = a+b

    Mu_3d, Tau_3d = Mu.reshape(mcS, K, D), Tau.reshape(mcS, K, D)
    quad = (X[None,:,None,:] - Mu_3d[:, None, :, :])**2/(np.expm1(Tau_3d[:, None, :, :]**2) + 1.)
    t2 = -0.5* np.sum(quad, axis = 3)

    log_llh = np.sum(   logsumexp(t1[:,None, :] + t2, axis = 2), axis = 1)
    return log_llh


def log_posterior(param, X, K, alpha=1):
    return log_prior(param, X, K, alpha) + log_llh(param, X, K)


def prior_sample(Sample_size, K, D, alpha=1):
    Mu = np.random.randn(K*D*Sample_size).reshape(Sample_size, K*D)
    Lmd = np.log1p(np.random.gamma(alpha, 1, K*Sample_size) - 1.).reshape(Sample_size, K)
    Tau = np.random.randn(K*D*Sample_size).reshape(Sample_size, K*D)
    return np.hstack((Mu, Lmd, Tau))


def syn_lpdf(param):
    K = 3
    D = 2
    syn_dat = np.load( '../data/syndat_gmm.npy')
    X = syn_dat[:, :2]
    return log_posterior(param, X, K)


def syn_pdf(param):
    # add 1500 to avoid underflow, won't affect the results
    lpdf = syn_lpdf(param) + 1500
    return np.exp(lpdf)


def real_lpdf(param):
    dat = np.load('../data/shapley_proc.npy')
    K = 7
    return log_posterior(param, dat, K)


def real_pdf(param):
    lpdf = real_lpdf(param)/2 + 300 #tempered posterior
    return np.exp(lpdf)


if __name__ == '__main__':
    # Generate psuedo data
    from sklearn.datasets.samples_generator import make_blobs
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    dat_gmm = np.hstack((X, y_true[:, None]))
    data_path = '../data/syndat_gmm.npy'
    np.save(data_path, dat_gmm)

