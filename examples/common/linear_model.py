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


if __name__ == '__main__':
    param = np.array([
        [0, 1, 0, 0],
        [1, 0, 5, 5],
        [1, 0, 5, 5],
        [0, 1, 0, 0]
    ])

    dat = gen_synthetic(10)
    X = dat[:, 1:]
    Y = dat[:, 0]

    print(lpdf_syn(param))
    def lpdf(theta): return log_posterior(theta, X, Y)
    def pdf(theta): return np.exp(lpdf(theta))
    from autograd import grad

    gg = grad(lpdf_fish)
    print(gg(np.arange(8)))


# if __name__=='__main__':
#   #tunning all alg here
#   sys.path.insert(1, os.path.join(sys.path[0], '../..'))
#   from VI.MAP import *
#   pdf = lambda x: np.exp(lpdf_fish(x)/1e2)
#   print(pdf(np.ones(64).reshape(8,8)))
#   # alpha= 0.1
#   # lrt = lambda itr: 5/(1+ itr)
#   # alpha0 = np.random.randn()
#   # lmd0 = np.random.randn()
#   # beta0 = np.random.randn(6)*np.exp(-0.5*alpha)
#   # init = np.hstack((beta0, alpha0, lmd0 ))
#   # mu0 = smooth_MAP(init,pdf ,lrt, alpha, num_iters= 100000)
#   # mu0  = MAP_est(np.ones(8), lpdf_fish, lrt)
#   # f = lambda x: lpdf_fish(x)/159
#   # G = grad(f)
#   # print(G(np.atleast_1d(np.ones(8))))


#   # ###
#   # import matplotlib.pyplot as plt
#   # f = lambda x: -lpdf_fish(x)/159
#   # MC = np.load('/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/MCsample.npy')
#   # mcmc = MC[-5000:]
#   # mc_mean  = np.mean(mcmc, axis = 0)
#   # x = np.linspace(-2, 2, 100)
#   # xxx = np.hstack((x[:, None] , np.tile(np.zeros(7), (100, 1))))
#   # y = f(xxx)
#   # plt.plot(x,y)
#   # plt.show()


#   from VI.csvi import *
#   from VI.svi import *
#   vi_lrt = lambda i: .1/(1+ i)
#   alpha0 = np.random.randn()
#   lmd0 = np.random.randn()
#   beta0 = np.random.randn(6)*np.exp(-0.5*alpha0)
#   mu0 = np.hstack((beta0, alpha0, lmd0 ))
#   L0 = np.identity(8)/np.sqrt(159)
#   init_val,_ = flatten([mu0, L0])
#   x = csvi_adam(init_val, 159, lpdf_fish, vi_lrt, 1000000)
