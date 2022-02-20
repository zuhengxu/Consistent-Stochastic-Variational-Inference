import scipy.misc
import scipy.special
from autograd.extend import primitive, defvjp
import autograd.numpy as np
from autograd.numpy.numpy_vjps import repeat_to_match_shape


# autodiff for logsumexp
logsumexp = primitive(scipy.special.logsumexp)


def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    shape, dtype = np.shape(x), np.result_type(x)

    def vjp(g):
        g_repeated,   _ = repeat_to_match_shape(
            g,   shape, dtype, axis, keepdims)
        ans_repeated, _ = repeat_to_match_shape(
            ans, shape, dtype, axis, keepdims)
        return g_repeated * b * np.exp(x - ans_repeated)
    return vjp


defvjp(logsumexp, make_grad_logsumexp)


# map the nan elelment of 'a' to the correpsponding value of 'b'
def nan_clean(a, b):
    def replace_nan(x, y):
        if np.isnan(x):
            x = y
        return x
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    aa = np.array(list(map(replace_nan, a, b)))
    return aa



def de_flatten(x):
    x = np.atleast_1d(x)
    dim = int(np.sqrt(np.size(x)))
    mu = x[0:dim]
    L = np.tril(x[dim:].reshape(dim, dim))
    return dim, mu, L


# map non-pos diag indices of L to a specific value a
def map_neg_diag(L, a):
    dim = L.shape[0]
    di = np.diag_indices(dim)  # creat diagonal indeces
    L_diag = L[di]
    L_diag[L_diag <= 0] = a

    L[di] = L_diag
    return L

def sym_inv(H):
    u, s, v = np.linalg.svd(H)
    Hinv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose()))
    return Hinv 


# update rule of adam
def adam_update(i, x, g, m, v, lrt, b1=0.9, b2=0.999, eps=1e-8):
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))    # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    x1 = x - lrt*mhat/(np.sqrt(vhat) + eps)
    return m, v, x1, x

def sgd_update(i, x, g, lrt):
    x1 = x- lrt*g
    return x1, x 


def backtracking_update(x, g, f, t = 1, b = 0.5):
    # choose step size
    while ( f(x - t*g) > (f(x) - 0.5*t*np.sum(g**2.0)) ):
        t *= b  
    x1 = x - t*g 
    # print(t)
    t = 10.0*t 
    return x1, t

# f = lambda x: np.sum((x*np.array([100, 0.1]))**2)
# g = grad(f)

# x = 100.0*np.ones(2)
# for i in range(10000):
#     G = g(x)
#     x = backtracking_update(x,G,f)
#     print(x)





