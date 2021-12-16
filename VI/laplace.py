from examples.common.elbo import *
import autograd.numpy as np
from autograd import grad, hessian
from autograd.misc import flatten
from .util import *






# consistent laplace approximation to logp at SMAP: mu0
# x0 = flatten[mu0, L0]
def cs_laplace(x0, log_post_pdf, learning_rate, num_iters, Newton_adjust = True):
    
    dim, mu0, L0 = de_flatten(x0)
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
        
        if np.isnan(mu).any():
            print("yes")
            mu = nan_clean(mu, mu0)

        if i % 1000 == 0:
            print(mu)

    H = h_logp(mu)
    Hinv = np.linalg.inv(H[0, :,:])
    G = g_logp(mu)
    Mean = mu - np.dot(Hinv, G) if Newton_adjust else mu
    L = np.linalg.cholesky(Hinv)
    x, _ = flatten([Mean, L])     

    return x 




def cs_laplace_1d(x0, log_post_pdf, learning_rate, num_iters, Newton_adjust = True):

    dim, mu0, L0 = de_flatten(x0)
    # scale log_post_pdf with N
    def f(x) : return -1.0*log_post_pdf(x)
    g_logp = grad(f)
    h_logp = hessian(f)

    mu = np.atleast_1d(mu0)
    # m_mu,v_mu = np.zeros(len(mu)), np.zeros(len(mu))
    # move around smap to make sure Hessian pd
    for i in range(num_iters):
        mu_grad = g_logp(mu)
        m_mu, v_mu, mu = adam_update( i, mu, mu_grad, m_mu, v_mu, learning_rate(i))

        
        if np.isnan(mu).any():
            print("yes")
            mu = nan_clean(mu, mu0)
        if i% 1000 == 0:
            print(mu)

    H = h_logp(mu)
    G = g_logp(mu)
    var = 1.0/(H + 1e-10) 
    Mean = mu - var*G if Newton_adjust else mu
    sd = np.sqrt(var)
    x = np.append(Mean, sd)

    return x

