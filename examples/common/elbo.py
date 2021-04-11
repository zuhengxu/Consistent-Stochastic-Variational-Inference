import autograd.numpy as np 

## gaussian entropy and ELBO 1d

def GVB_ELBO(unorm_lpdf, mu, sd, num_samples = 1000):
    samples = np.random.randn(num_samples)*sd + mu
    gs_entropy = np.log(sd) + np.log(2*np.pi)/2 + 0.5
    return gs_entropy + np.mean(unorm_lpdf(samples))


# multivariate ELBO estimation: x,_ =  flatten([mu, L])
def multi_ELBO(lpdf, x, num_samples = 1000):
    x = np.atleast_1d(x)
    dim = int(np.sqrt(np.size(x)))
    mean  = x[0:dim]
    L = np.tril(x[dim:].reshape(dim,dim))
    d = np.size(mean)
    gs_entropy = d*(1+ np.log(2*np.pi))/2  + np.log(np.linalg.det(L))
    samples = np.dot(L,np.random.randn(d, num_samples) ).T + np.tile(mean , (num_samples,1))
    est = lpdf(samples).mean()
    return gs_entropy + est


