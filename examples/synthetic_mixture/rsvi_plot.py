import os, sys,glob,pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#create figure folder if it does not exist
if not os.path.exists('figures/'):
        os.mkdir('figures/')

#result dir
res_dir = os.path.join(sys.path[0], 'results/VI_results/')

# ## color palette
# col_pal = { 'RSVI':'plum' ,'RSVI_OPT':'olive'}

# create dataframe of elbo
df0 = pd.read_csv( '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/synthetic_mixture/results/VI_results/results_SVI.csv' ) 
df1 = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/synthetic_mixture/results/rsvi/regRSVI.csv')
# df2 = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/VI_results/SYN_RSVI_SMAP_adamInd.csv')

df0 = (df0.assign(lmd = 0).rename(columns= {'lmd': 'Regularization', 'elbo': 'ELBO'}))[['Regularization', 'ELBO']]
df1 = df1.rename(columns = {'lmd': 'Regularization', 'elbo': 'ELBO'})[['Regularization', 'ELBO']]
df_syn = pd.concat([df0, df1])

f, ax = plt.subplots()
ax = sns.violinplot(x = 'Regularization', y = 'ELBO',data = df_syn,
                    scale = 'count', inner = 'stick',bw = 0.03, linewidth=0.3, gridsize=1000 )
plt.xlabel('Regularization', fontsize = 18)
plt.ylabel('ELBO', fontsize = 18)
plt.title('Regularized SVI', fontsize = 18)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize = 12, ncol = 4)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
f.savefig('figures/rsvi.png',bbox_inches='tight',dpi = 500)

######################3
###optimal init 
#################
df3 = pd.read_csv( '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/synthetic_mixture/results/VI_results/results_SVI_OPT.csv' ) 
df2 = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/synthetic_mixture/results/rsvi/regRSVI_OPT.csv')
# df2 = pd.read_csv('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/Gaussian_mixture/results/VI_results/SYN_RSVI_SMAP_adamInd.csv')

df3 = (df3.assign(lmd = 0).rename(columns= {'lmd': 'Regularization', 'elbo': 'ELBO'}))[['Regularization', 'ELBO']]
df2 = df2.rename(columns = {'lmd': 'Regularization', 'elbo': 'ELBO'})[['Regularization', 'ELBO']]
df_syn_opt = pd.concat([df3, df2])

f1, ax1 = plt.subplots()
ax1 = sns.violinplot(x = 'Regularization', y = 'ELBO',data = df_syn_opt,
                    scale = 'count', inner = 'stick',bw = 0.03, linewidth=0.3, gridsize=1000)
plt.xlabel('Regularization', fontsize = 18)
plt.ylabel('ELBO', fontsize = 18)
plt.title('Regularized SVI + SMAP', fontsize = 18)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize = 12, ncol = 4)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
f1.savefig('figures/rsvi_opt.png',bbox_inches='tight',dpi = 500)