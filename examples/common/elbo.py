from distutils.log import debug
import autograd.numpy as np
from scipy.stats import multivariate_normal as mvnorm
from VI.util import de_flatten


##################
# reverse KL
#####################
## ELBO 1d
def GVB_ELBO(unorm_lpdf, mu, sd, num_samples = 1000):
    samples = np.random.randn(num_samples)*sd + mu
    gs_entropy = np.log(sd) + np.log(2*np.pi)/2 + 0.5
    return gs_entropy + np.mean(unorm_lpdf(samples))


# multivariate ELBO estimation: x,_ =  flatten([mu, L])
def multi_ELBO(lpdf, x, num_samples = 1000):
    x = np.atleast_1d(x)
    d, mean, L = de_flatten(x)
    # print(x[dim:].reshape(dim,dim))
    gs_entropy = d*(1+ np.log(2*np.pi))/2  + np.sum(np.log(np.diagonal(L))) # sum the log diag
    samples = np.dot(L,np.random.randn(d, num_samples) ).T + np.tile(mean , (num_samples,1))
    # print(samples.shape)
    est = lpdf(samples).mean()
    return gs_entropy + est
    # return est

#########################
# forward KL
#########################
def fwd_KL(lpdf, x, post_samples):
    d, mu, L = de_flatten(x)
    Sigma = np.dot(L, L.T) 
    rv = mvnorm(mu, Sigma)
    log_ratio = lambda x: lpdf(x) - rv.logpdf(x)
    est = log_ratio(post_samples).mean()
    return est
