# from sklearn.datasets.samples_generator import make_blobs
from VI.util import *
import autograd.numpy as np
import os
import sys
sys.path.append(
    "/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code")


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


# def log_likelihood(param, X, K):
#     param = np.atleast_2d(param)
#     mcS = param.shape[0]
#     N = X.shape[0]
#     D = X.shape[1]
#     Mu = param[:, :K*D]
#     Lam = param[:, K*D:K*D + K]
#     Tau = param[:, -K*D:]

#     a = Lam - logsumexp(Lam, axis=1)[:, None]
#     term1 = np.tile(a, (N, 1, 1))
#     term2 = np.tile(-0.5*(np.sum(Tau.reshape(mcS, K, D)**2, axis=2)), (N, 1, 1))

#     Mu_3d, Tau_3d = np.swapaxes(Mu.reshape(mcS, K, D), 0, 1),  np.swapaxes(Tau.reshape(mcS, K, D), 0, 1)
#     Mu_3dstack, Tau_3dstack = np.tile(Mu_3d, (N, 1, 1)), np.tile(Tau_3d, (N, 1, 1))
#     X_3d = np.swapaxes(np.tile(X, (mcS, 1, 1)), 0, 1)
#     X_3dstack = np.repeat(X_3d, K, 0)
#     qua_stack = (X_3dstack - Mu_3dstack)**2/np.exp(Tau_3dstack**2)
#     qua = np.sum(qua_stack, axis=2).T
#     term3 = - 0.5 * np.swapaxes(qua.reshape(mcS, N, K), 0, 1)

#     log_likelihood = np.sum(logsumexp(term1 + term2 + term3, axis=2), axis=0)
#     return log_likelihood


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
    syn_dat = np.load(
        '/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/syndat_gmm.npy')
    X = syn_dat[:, :2]
    return log_posterior(param, X, K)


def syn_pdf(param):
    # add 1500 to avoid underflow, won't affect the resutls
    lpdf = syn_lpdf(param) + 1500
    return np.exp(lpdf)


def real_lpdf(param):
    dat = np.load(
        '/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/shapley_proc.npy')
    K = 7
    return log_posterior(param, dat, K)


def real_pdf(param):
    lpdf = real_lpdf(param)/2 + 300 #tempered posterior
    return np.exp(lpdf)


if __name__ == '__main__':
    # Generate psuedo data
    from sklearn.datasets.samples_generator import make_blobs
    X, y_true = make_blobs(n_samples=400, centers=4,
                           cluster_std=0.6, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    dat_gmm = np.hstack((X, y_true[:, None]))
    data_path = '/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/syndat_gmm.npy'
    np.save(data_path, dat_gmm)

    # test posterior function
    K = 3
    param = np.ones(150).reshape(10, 15)
    syn_dat = np.load(
        '/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/syndat_gmm.npy')
    X = syn_dat[:, :2]
    from autograd import grad
    def lpdf(param): return log_posterior(param, X, 3)
    gg = grad(lpdf)
    gg(param[:, 1])
