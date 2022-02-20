from examples.common.elbo import *
from autograd.scipy import stats
import autograd.numpy as np
from scipy.stats import norm
from autograd import grad
from autograd.misc import flatten
from .util import *
from progress.bar import Bar

# svi: simple projected sgd
def svi(x0, N, log_post_pdf, learning_rate, num_iters):

    dim, mu, L = de_flatten(x0)
    #make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L,1.)
    # scale log_post_pdf with N
    f = lambda x: log_post_pdf(x)/N
    gobj = grad(f)
    x= np.atleast_1d(x0)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:,np.newaxis]).T[0]
        #grad for mu and L
        mu_grad = - gobj(zz)
        L_grad  = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z))
        #updates
        mu = mu - learning_rate(i)* mu_grad
        L = L - learning_rate(i)* L_grad
        L = map_neg_diag(L,1e-4)  # projection step: map non-pos idx of L to 1e-4
        # map nan value to init
        x,_ = flatten([mu, L])
        if np.isnan(np.sum(x)):
            x = nan_clean(x, x0)
            _, mu, L = de_flatten(x)
        #est elbo value
        if i%1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x


# svi with adam update
def svi_adam(x0, N, log_post_pdf, learning_rate, num_iters):
    bar = Bar('running csvi', max = num_iters)
    dim, mu, L = de_flatten(x0)
    #make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L,1.)
    # scale log_post_pdf with N
    f = lambda x: log_post_pdf(x)/N
    gobj = grad(f)
    x= np.atleast_1d(x0)
    m_mu, m_L  = np.zeros(len(mu)), np.zeros(L.shape)
    v_mu, v_L  = np.zeros(len(mu)), np.zeros(L.shape)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:,np.newaxis]).T[0]
        #grad for mu and L
        mu_grad = - gobj(zz)
        L_grad  = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z))
        # adam updates: m,v,x
        m_mu, v_mu, mu, mu_pre = adam_update(i, mu, mu_grad, m_mu, v_mu, learning_rate(i))
        m_L, v_L, L, L_pre = adam_update(i, L, L_grad, m_L, v_L, learning_rate(i))
        L = map_neg_diag(L,1e-4) #projection step
        # map nan value to init
        x,_ = flatten([mu, L])
        x_pre, _ = flatten([mu_pre, L_pre])
        if np.isnan(np.sum(x)):
            x = nan_clean(x, x_pre)
            _, mu, L = de_flatten(x)
        
        bar.next()
        #est elbo value
        if i%1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x



# svi with variance regularization 
def rsvi(x0, N, log_post_pdf, learning_rate, num_iters, lmd):

    dim, mu, L = de_flatten(x0)
    #make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L,1.)
    # scale log_post_pdf with N
    f = lambda x: log_post_pdf(x)/N
    gobj = grad(f)
    x= np.atleast_1d(x0)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:,np.newaxis]).T[0]
        #grad for mu and L
        mu_grad = - gobj(zz)
        L_grad  = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z)) + 2.0*lmd* (L - np.identity(L.shape[0]))
        #updates
        mu = mu - learning_rate(i)* mu_grad
        L = L - learning_rate(i)* L_grad
        L = map_neg_diag(L,1e-4)  # projection step: map non-pos idx of L to 1e-4
        # map nan value to init
        x,_ = flatten([mu, L])
        if np.isnan(np.sum(x)):
            x = nan_clean(x, x0)
            _, mu, L = de_flatten(x)
        #est elbo value
        if i%1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x
# lmd is the regularization constant 
def rsvi_adam(x0, N, log_post_pdf, learning_rate, num_iters, lmd):
    dim, mu, L = de_flatten(x0)
    #make sure initial L invertible, map non-pos diag idx to 1
    L = map_neg_diag(L,1.)
    # scale log_post_pdf with N
    f = lambda x: log_post_pdf(x)/N
    gobj = grad(f)
    x= np.atleast_1d(x0)
    m_mu, m_L  = np.zeros(len(mu)), np.zeros(L.shape)
    v_mu, v_L  = np.zeros(len(mu)), np.zeros(L.shape)

    for i in range(num_iters):
        Z = np.random.randn(dim)
        zz = mu + np.dot(L, Z[:,np.newaxis]).T[0]
        #grad for mu and L
        mu_grad = - gobj(zz)
        # add regularization here
        L_grad  = - np.diag(1/np.diag(L))/N + np.tril(np.outer(mu_grad, Z)) + 2.0*lmd* (L - np.identity(L.shape[0]))
        # adam updates: m,v,x.
        m_mu, v_mu, mu = adam_update(i, mu, mu_grad, m_mu, v_mu, learning_rate(i))
        m_L, v_L, L = adam_update(i, L, L_grad, m_L, v_L, learning_rate(i))
        L = map_neg_diag(L,1e-4) #projection step
        # map nan value to init
        x,_ = flatten([mu, L])
        if np.isnan(np.sum(x)):
            x = nan_clean(x, x0)
            _, mu, L = de_flatten(x)
        #est elbo value
        if i%1000 == 0:
            print(multi_ELBO(log_post_pdf, x))
    return x



