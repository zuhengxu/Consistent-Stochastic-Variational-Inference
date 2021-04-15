import autograd.numpy as np
import pandas as pd
import os
import sys
from autograd import grad


def gen_synthetic(n):
    X = np.random.randn(n)
    X = np.hstack((np.ones(n)[:, np.newaxis], X[:, np.newaxis]))
    y = np.random.randn(n)*2. + np.dot(X, np.array([1., 2.]))
    return np.hstack((y[:, np.newaxis], X))


def log_posterior(param, X, Y):
    param = np.atleast_2d(param)
    k = param.shape[0]
    beta = np.atleast_2d(param[:, :-2])
    alpha = param[:, -2]
    lambd = param[:, -1]
    n = X.shape[0]
    d = beta.shape[1]
    Y = np.atleast_1d(Y)[:, np.newaxis]

    log_llh = n/2*lambd - 0.5*np.exp(lambd)*np.diag(np.dot((np.tile(
        Y, (1, k)) - np.dot(X, beta.T)).T, (np.tile(Y, (1, k)) - np.dot(X, beta.T))))
    log_priors = d/2*alpha - 0.5 * np.exp(alpha)*np.diag(np.dot(beta, beta.T)) - 0.5*(alpha**2 + lambd**2)
    return log_llh+log_priors


def lpdf_syn(param):
    syn_dat_path = os.path.join(sys.path[0], '../data/linreg_syn.csv')
    dat = pd.read_csv(syn_dat_path)
    data = dat.to_numpy()
    Y = data[:, 0]
    X = data[:, 1:]
    return log_posterior(param, X, Y)


# fish data set with features and Y normialized into [0,1]
def lpdf_fish(param):
    dat_path = os.path.join(sys.path[0], '../data/Fish.csv')
    dat = pd.read_csv(dat_path).iloc[:, 1:]
    data = dat.to_numpy()
    Y = data[:, 0]
    Y_proc = (Y-Y.mean()) / Y.std()
    X_raw = data[:, 1:]
    m = np.mean(X_raw, axis=0)
    sd = np.std(X_raw, axis=0)
    X_proc = (X_raw - m)/sd
    Z = np.hstack((np.ones(X_proc.shape[0])[:, np.newaxis], X_proc))
    return log_posterior(param, Z, Y_proc)


