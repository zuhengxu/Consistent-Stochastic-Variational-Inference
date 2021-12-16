import autograd.numpy as np
from pandas.io.parsers import read_csv
from scipy.stats import norm
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from seaborn import palettes
from VI.util import de_flatten
### GMM syn (why lap so bad)
results_dir = os.path.join('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture', 'results/VI_results/')
df_csl = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/VI_results/SYN_CSL_SMAP_adamInd.csv')
df_csvi = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/VI_results/SYN_CSVI_adam_SMAP_adamInd.csv')
df_svi = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/VI_results/SYN_SVI_adam_PriorInd.csv')
df_init = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/initials/SYN_SMAP_adamInd.csv')

res_csl = np.array(df_csl)[0, 1:]
res_csvi = np.array(df_csvi)[0, 1:]
res_svi = np.array(df_svi)[0, 1:]
init = np.array(df_init)[0]

d, mu_csl, L_csl = de_flatten(res_csl)
d, mu_csvi, L_csvi = de_flatten(res_csvi)
d, mu_svi, L_svi = de_flatten(res_svi)

d, smap, L_init = de_flatten(init) 



def lpdf_slice(x):
    a = np.concatenate((mu_csvi[0:2], np.array([x]), mu_csvi[3:]))
    return -syn_lpdf(a)

x = np.linspace(-50, 50, 1000)
plt.plot(x, [lpdf_slice(a) for a in x])
plt.show()
