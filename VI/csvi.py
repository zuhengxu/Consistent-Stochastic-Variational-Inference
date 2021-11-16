from examples.common.elbo import *
import autograd.numpy as np
from autograd import grad, hessian
from autograd.misc import flatten
from .util import *
# from util import *



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



# consistent laplace approximation to logp at SMAP: mu0
def cs_laplace(mu0, log_post_pdf, learning_rate, num_iters):
    
    # scale log_post_pdf with N
    def f(x) : return -1.0*log_post_pdf(x)
    g_logp = grad(f)
    h_logp = hessian(f)

    mu = np.atleast_1d(mu0)
    m_mu,v_mu = np.zeros(len(mu)), np.zeros(len(mu))

    # move around smap to make sure Hessian pd
    for i in range(num_iters):
        mu_grad = g_logp(mu)
        m_mu, v_mu, mu = adam_update( i, mu, mu_grad, m_mu, v_mu, learning_rate(i))
        print(mu)
        
        if np.isnan(mu).any():
            print("yes")
            mu = nan_clean(mu, mu0)

    H = h_logp(mu)
    Hinv = np.linalg.inv(H[0, :,:])
    Mean = -np.dot(Hinv, mu)
    # what if Hinv not pd 
    L = np.linalg.cholesky(Hinv)
    x, _ = flatten([Mean, L])     
    print(x)
    return x 

def cs_laplace_1d(mu0, log_post_pdf, learning_rate, num_iters):

    # scale log_post_pdf with N
    def f(x) : return -1.0*log_post_pdf(x)
    g_logp = grad(f)
    h_logp = hessian(f)

    mu = np.atleast_1d(mu0)
    m_mu,v_mu = np.zeros(len(mu)), np.zeros(len(mu))
    
    # move around smap to make sure Hessian pd
    for i in range(num_iters):
        mu_grad = g_logp(mu)
        m_mu, v_mu, mu = adam_update( i, mu, mu_grad, m_mu, v_mu, learning_rate(i))
        print(mu)
        
        if np.isnan(mu).any():
            print("yes")
            mu = nan_clean(mu, mu0)

    H = h_logp(mu)
    var = 1.0/(H + 1e-10) 
    Mean = -var*mu 
    sd = np.sqrt(var)
    x = np.append(Mean, sd)

    return x




if __name__ == '__main__':
    # testing on synthetic Gaussian mixture 
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

    def syn_lpdf(param):
        K = 3
        D = 2
        syn_dat = np.load( '../examples/data/syndat_gmm.npy')
        X = syn_dat[:, :2]
        # print(X)
        return log_posterior(param, X, K)

    # h_logp = hessian(syn_lpdf)
    # theta = np.ones(12)
    # A = h_logp(theta)[0, :, :]
    # g = grad(syn_lpdf)
    # b = g(theta)
    # # u, s, v  =np.linalg.svd(A)
    # # print(np.transpose(A))
    # print(np.tril(A))
    # # x = np.dot(np.linalg.inv(A), b)
    # # print(np.diagonal(np.transpose(A)))
    # # print(s)
    # # print(sym_inv(A[0,:,:]))
    # # print(np.shape(A[0,:, :]))
    cs_laplace(np.ones(12), syn_lpdf, lambda i: 0.005, 5000)