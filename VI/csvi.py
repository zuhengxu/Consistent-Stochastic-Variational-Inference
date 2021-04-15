from examples.common.elbo import *
import autograd.numpy as np
from autograd import grad
from autograd.misc import flatten
from .util import *



# csvi
def csvi(x0, N, log_post_pdf, learning_rate, num_iters):

    dim, mu, L = de_flatten(x0)
    # make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L, 1.)
    # scale log_post_pdf with N
    def f(x): return log_post_pdf(x)/N
    gobj = grad(f)
    x = np.atleast_1d(x0)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:, np.newaxis]).T[0]  # modify with extra factor
        # grad for mu and scaled grad for L
        mu_grad = - gobj(zz)
        di = np.diag_indices(dim)  # creat diagonal indeces
        # modify with extra factor
        L_grad = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z))
        L_grad[di] = (N*L[di] / (1 + N*L[di])) * L_grad[di]
        # updates
        mu = mu - learning_rate(i) * mu_grad
        L = L - learning_rate(i) * L_grad
        L = map_neg_diag(L, learning_rate(i))  # projection step

        # map nan value to init
        x, _ = flatten([mu, L])
        if np.isnan(np.sum(x)):
            x = nan_clean(x, x0)
            _, mu, L = de_flatten(x)
        # est elbo value
        if i % 1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x


# csvi using adam update
def csvi_adam(x0, N, log_post_pdf, learning_rate, num_iters):

    dim, mu, L = de_flatten(x0)
    # make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L, 1.)
    # scale log_post_pdf with N
    def f(x): return log_post_pdf(x)/N
    gobj = grad(f)
    x = np.atleast_1d(x0)
    m_mu, m_L = np.zeros(len(mu)), np.zeros(L.shape)
    v_mu, v_L = np.zeros(len(mu)), np.zeros(L.shape)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:, np.newaxis]).T[0]  # modify with extra factor
        # grad for mu and scaled grad for L
        mu_grad = - gobj(zz)
        di = np.diag_indices(dim)  # creat diagonal indeces
        # modify with extra factor
        L_grad = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z))
        L_grad[di] = (N*L[di] / (1 + N*L[di])) * L_grad[di]
        # updates
        m_mu, v_mu, mu = adam_update( i, mu, mu_grad, m_mu, v_mu, learning_rate(i))
        m_L, v_L, L = adam_update(i, L, L_grad, m_L, v_L, learning_rate(i))
        L = map_neg_diag(L, learning_rate(i))  # projection step

        # map nan value to init
        x, _ = flatten([mu, L])
        if np.isnan(x).any():
            x = nan_clean(x, x0)
            _, mu, L = de_flatten(x)
        # est elbo value
        if i % 1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x
