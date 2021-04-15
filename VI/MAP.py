import autograd.numpy as np
from autograd import grad
from .util import *



def smth_grd(lpdf, alpha, mcS, theta):
    d = np.size(theta)
    Z = np.random.randn(mcS, d)
    logpi = lpdf(theta - np.sqrt(alpha)*Z)
    G = np.sign(Z)* (np.expm1( np.log1p(np.abs(Z) - 1.)  + logpi[:,None] -logsumexp(logpi)  + 1.))
    return np.sum(G, axis = 0)


# smooth map estimation, using monte carlo gradient
def smooth_MAP(x0, lpdf, learning_rate, alpha, mcS=100, num_iters=10000):

    x = np.atleast_1d(x0)
    def G(x): return smth_grd(lpdf, alpha, mcS, x)

    for i in range(num_iters):
        g = G(x)
        x = x - learning_rate(i)*g
        # if nan value encounter: map back to init_val
        x = nan_clean(x, x0)
        if i % 100 == 0:
            print(lpdf(x), x)
    return x

# smooth map estimation but with adam update
def SMAP_adam(x0, lpdf, learning_rate, alpha, mcS=100, num_iters=10000):

    x = np.atleast_1d(x0)
    def G(x): return smth_grd(lpdf, alpha, mcS, x)
    m, v = np.zeros(len(x)), np.zeros(len(x))

    for i in range(num_iters):
        g = G(x)
        m, v, x = adam_update(i, x, g, m, v, learning_rate(i))
        # if nan value encounter: map back to init_val
        if np.isnan(x).any():
            x = nan_clean(x, x0)
        if i % 100 == 0:
            print(lpdf(x), x)
    return x



# 1d smooth map estimation(faster), using monte carlo gradient
def smooth_1d_MAP(x0, lpdf, learning_rate, alpha,  mcS=100, num_iters=10000):

    def smth_grd(lpdf, alpha, k, theta):
        Z = np.random.randn(k)
        logpi = lpdf(theta - np.sqrt(alpha)*Z)
        G = np.sign(Z)* (np.expm1( np.log1p(np.abs(Z) - 1.)  + logpi -logsumexp(logpi)  + 1.))
        return np.sum(G)

    i = 0
    x = np.atleast_1d(x0)
    def G(x): return smth_grd(lpdf, alpha, mcS, x)

    for i in range(num_iters):
        g = G(x)
        x = x - learning_rate(i)*g
        x[np.isnan(x)] = x0

        if i % 1000 == 0:
            print(i, x, g)
    return x

