import autograd.numpy as np
from autograd import grad
from .util import *


# def MAP_est(x0, f, learning_rate, niters=10000):
#     G = grad(f)
#     x = np.atleast_1d(x0)

#     for i in range(niters):
#         g = G(x)
#         x = x + learning_rate(i)*g
#         if i % 10 == 0:
#             print(f(x))
#     return x


# map_lrt = lambda itr: 150/(1+ itr)
# map_est(np.array([(np.random.random() - 0.5)*100]), log_smth_gs_mix, map_lrt, 1000, 1e-5)

# a = np.ones((3,6))
# Z = np.random.randn(6, 3)
# np.sign(Z)
# np.abs(Z)

def smth_grd(lpdf, alpha, mcS, theta):
    d = np.size(theta)
    Z = np.random.randn(mcS, d)
    logpi = lpdf(theta - np.sqrt(alpha)*Z)
    G = np.sign(Z)* (np.expm1( np.log1p(np.abs(Z) - 1.)  + logpi[:,None] -logsumexp(logpi)  + 1.))
    return np.sum(G, axis = 0)


# def smth_grd1(unorm_pdf, alpha, mcS, theta):
#     theta = np.atleast_1d(theta)
#     dim = np.size(theta)
#     W = np.random.randn(mcS, dim)
#     denom = unorm_pdf(theta - np.sqrt(alpha)*W)
#     num = W*denom[:, np.newaxis]
#     g = num.mean(axis=0)/(np.sqrt(alpha)*denom.mean())
#     return g


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

# alpha= 0.1
# gaussian_mix = lambda x: np.exp(log_gs_mix(x))
# smth_map_lrt = lambda itr: 30/(1+ itr*(0.9))
# smooth_MAP(50, gaussian_mix, alpha, 10, smth_map_lrt, 1500)
