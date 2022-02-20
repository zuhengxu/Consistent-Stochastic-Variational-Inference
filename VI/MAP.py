import autograd.numpy as np
from autograd import grad
from .util import *
from progress.bar import Bar



def smth_grd(lpdf, alpha, mcS, theta):
    d = np.size(theta)
    Z = np.random.randn(mcS, d)
    logpi = lpdf(theta - np.sqrt(alpha)*Z)
    # print(logpi.shape)
    G = np.sign(Z)* np.exp( np.log(np.abs(Z))  + logpi[:,None] -logsumexp(logpi))
    return np.sum(G, axis = 0)/np.sqrt(alpha)

# a = np.random.randn(10)
# smth_grd(real_lpdf, 1., 10000, np.zeros(8))

# smooth map estimation, using monte carlo gradient
def smooth_MAP(x0, lpdf, learning_rate, alpha, mcS=100, num_iters=10000):
    bar = Bar('running Smooth MAP', max = num_iters/1000)

    x = np.atleast_1d(x0)
    def G(x): return smth_grd(lpdf, alpha, mcS, x)

    for i in range(num_iters):
        g = G(x)
        x = x - learning_rate(i)*g
        # if nan value encounter: map back to init_val
        x = nan_clean(x, x0)
        if i % 1000 == 0:
            bar.next()
    return x




# smooth map estimation but with adam update
def SMAP_adam(x0, lpdf, learning_rate, alpha, mcS=100, num_iters=10000):
    bar = Bar('running Smooth MAP (adam)', max = num_iters/100)
    x = np.atleast_1d(x0)
    def G(x): return smth_grd(lpdf, alpha, mcS, x)
    m, v = np.zeros(len(x)), np.zeros(len(x))

    for i in range(num_iters):
        g = G(x)
        m, v, x, x_pre = adam_update(i, x, g, m, v, learning_rate(i))
        # if nan value encounter: map back to init_val
        if np.isnan(x).any():
            x = nan_clean(x, x_pre)
        if i % 100 == 0:
            bar.next()
            # print("\n", "mu" ,x)
            # print("\n", "grad", G(x))
    return x



# def smth_grd_ess(lpdf, alpha, mcS, theta):
#     d = np.size(theta)
#     Z = np.random.randn(mcS, d)
#     logpi = lpdf(theta - np.sqrt(alpha)*Z)
#     G = np.sign(Z)* (np.expm1( np.log1p(np.abs(Z) - 1.)  + logpi[:,None] -logsumexp(logpi)  + 1.))
#     pi = np.exp(logpi)
#     ess = np.sum(pi)**2.0 / np.sum(pi**2)
#     return np.sum(G, axis = 0), ess

# def SMAP_adam_cont(x0, lpdf, learning_rate, alpha, m0, v0, mcS=100, num_iters=10000):
#     # bar = Bar('running Smooth MAP (adam)', max = num_iters)
#     x = np.atleast_1d(x0)
#     def G(x): return smth_grd_ess(lpdf, alpha, mcS, x)
#     m, v = m0, v0 

#     for i in range(num_iters):
#         g, ess = G(x)
#         m, v, x, x_pre = adam_update(i, x, g, m, v, learning_rate(i))
#         # if nan value encounter: map back to init_val
#         if np.isnan(x).any():
#             print("\n", "Nan Appear")
#             x = nan_clean(x, x_pre)
#         # bar.next()
#         if i % 1000 == 0:
#             print('\n', i, '/' ,num_iters)
#             print("\n", "mu" ,x)
#             print("\n", "grad", G(x))
#             # print("\n", "ess = ", ess)
#     return x, m, v


# x0 = prior_sample(1, 8, 0.01, 5)[0]
# x0 = np.array([0.5, 0.1, 0, 0, 0, 0, 0, 0])
# x100, m, v = SMAP_adam_cont(x0, real_lpdf, lambda iter: 0.0004, 0.01, np.zeros(8),np.zeros(8), 100, 300000)
# x1000, m1, v1= SMAP_adam_cont(x100, real_lpdf, lambda iter: 0.0005, 1, m, v, 500, 100000)

fh = lambda iter: 0.5/(1000 + iter**0.5)

# 1d smooth map estimation(faster), using monte carlo gradient
def smooth_1d_MAP(x0, lpdf, learning_rate, alpha,  mcS=100, num_iters=10000):
    bar = Bar('running Smooth MAP', max = num_iters)
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

        bar.next()
        if i % 1000 == 0:
            print(i, x, g)
    return x

