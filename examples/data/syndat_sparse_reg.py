import autograd.numpy as np

D = 5 # number of features
N = 10 # number of obs
X = np.random.randn(N, D) # covariates
#generate true coef
beta_true = np.hstack((np.random.randn(1)+5, np.zeros(D -1)))
Y = np.random.randn(N)*0.5 + np.dot(X, beta_true)
dat_sparse_reg = np.hstack((Y[:, None], X))
data_path = 'data/syndat_sparse_reg.npy'
np.save(data_path, dat_sparse_reg)


