from autograd import grad
import os, sys
import autograd.numpy as np
from autograd.scipy import stats
from VI.util import *



def toy_unorm_lpdf(x, sample_size):

    n = sample_size
    lw = np.log(np.array([0.2,0.2,0.2, 0.2, 0.2]))
    lpdf = np.hstack((stats.norm.logpdf(x, -1, 0.15)[:, np.newaxis],
            stats.norm.logpdf(x, 1, 0.1)[:, np.newaxis],
            stats.norm.logpdf(x, -4, 0.3)[:, np.newaxis],
            stats.norm.logpdf(x, 4, 0.3)[:, np.newaxis],
            stats.norm.logpdf(x, -8, 0.1)[:, np.newaxis] ))

    log_prior = logsumexp(lw + lpdf, axis= 1)

    r = np.random.normal(size = n)*np.sqrt(10) + 3
    llh =  -(np.sum(r**2)- 2* np.sum(r)*x + n*(x**2))/10000
    # likelihood = (np.exp(-(x-5)**2/(2*2))/np.sqrt(2*np.pi*2))**n
    #

    return log_prior + llh


