from scipy.stats import norm
import os, sys,glob,pickle
import matplotlib.pyplot as plt
import numpy as npp
import seaborn as sns
import pandas as pd
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from examples.common.elbo import *
from examples.common.sparse_reg_model import *
# set up progress bar
# from progress.bar import Bar
# bar = Bar('Computing KL', fill='#', suffix='%(percent).1f%% - %(eta)ds')



###########################
# compare Forward KL for real dataset  
###########################
# read stan trace to get post samples
stan_result_path = 'results/stan/'
# stan_result_path = '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan'
trace = pickle.load(open(os.path.join(stan_result_path, 'sr_real_trace.pkl'), 'rb'))
df_beta = pd.DataFrame(trace['beta'])
df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
# thining MCMC samples every 10
post_samples = np.array(df_beta[df_beta.index % 10 == 0])

# useful functions
def df_to_array(D): 
    return npp.array(D)[:,1:]
compute_kl = lambda x: fwd_KL(real_lpdf, x, post_samples)

# read resutls
df_cs = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamInd.csv')
df_csrand = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamRandom.csv')
df_svi = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorRandom.csv')
df_svi_ind = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorInd.csv')
df_lap = pd.read_csv('results/VI_results/REAL_CSL_PriorRandom.csv')
df_cl = pd.read_csv('results/VI_results/REAL_CSL_SMAP_adamInd.csv')


# T = df_to_array(df_cl)
# kl = npp.apply_along_axis(compute_kl, 1, T)

##################################
# computing fwd KL
###################################

# initialize an empty dict
dict = {}
for alg, df in zip(['CSVI(adam)', 'CSVI(adam)_Rand', 'CSL', 'SVI(adam)_Ind', 'SVI(adam)_Rand'],
                    [df_cs, df_csrand, df_cl, df_svi_ind, df_svi]):
# for alg, df in zip(['CSVI(adam)', 'CSVI(adam)_Rand', 'Laplace', 'SVI(adam)_Ind', 'SVI(adam)_Rand'],
#                     [df_cs, df_csrand, df_lap, df_svi_ind, df_svi]):
    T = df_to_array(df)
    kl = npp.apply_along_axis(compute_kl, 1, T)
    # print(kl)
    dict[alg] = npp.log10(kl)

real_kl = pd.DataFrame(dict)
df_real_kl = real_kl.melt(value_vars=['CSVI(adam)', 'CSVI(adam)_Rand','CSL','SVI(adam)_Ind', 'SVI(adam)_Rand'], 
# df_real_kl = real_kl.melt(value_vars=['CSVI(adam)', 'CSVI(adam)_Rand', 'Laplace', 'SVI(adam)_Ind', 'SVI(adam)_Rand'], 
                        var_name='method', value_name= 'fwd_KL', ignore_index = True) 
df_real_kl.replace('CSL', 'CLA', inplace= True)


############################
# Voilin plot for forward KL (REAL)
#############################
# make plot
f2, ax2 = plt.subplots()
# sns.set_theme(style="whitegrid")
ax2 = sns.violinplot(x = 'method', y = 'fwd_KL',data = df_real_kl, 
                    scale = 'count', inner = 'stick', bw = 0.1,
                    order = ['CSVI(adam)', 'CLA'])
                    # order = ['CSVI(adam)', 'CSVI(adam)_Rand', 'Laplace' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'])
plt.xlabel('')
plt.ylabel('Log10 Foward KL', fontsize =18)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation  = 0)
# ax2.set_ylim(ymin = 0, ymax =100)
ax2.tick_params(axis="both", labelsize= 13)
f2.savefig('figures/sr_real_fwdkl.png',bbox_inches='tight',dpi = 500)
