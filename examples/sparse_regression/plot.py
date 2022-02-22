from pandas.io.parsers import read_csv
from scipy.stats import norm
import os, sys,glob,pickle
import matplotlib.pyplot as plt
import numpy as npp
from scipy.stats import multivariate_normal as mvnorm
import seaborn as sns
import pandas as pd
from seaborn import palettes
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from examples.common.results import *
from examples.common.plotting import *
from VI.util import de_flatten

###################
###################
### set up ########
###################
###################


#create figure folder if it does not exist
if not os.path.exists('figures/'):
        os.mkdir('figures/')

#result dir
results_dir = os.path.join(sys.path[0], 'results/VI_results/')


################################################
### p1. SYN elbo volin plot
################################################


# load results
CSVI_elbo = []
for file in glob.glob(results_dir + "SYN_CSVI_adam*.csv"):
    df_CSVI = pd.read_csv(file)
    CSVI_elbo.append(np.array(df_CSVI['0']))

SVI_elbo= []
for file in glob.glob(results_dir + "SYN_SVI_adam*.csv"):
        df_SVI = pd.read_csv(file)
        SVI_elbo.append(np.array(df_SVI['0']))

CSL_elbo= []
for file in glob.glob(results_dir + "SYN_CSL*.csv"):
        df_CSL = pd.read_csv(file)
        CSL_elbo.append(np.array(df_CSL['0']))

SYN_ELBO = {'CSVI' : CSVI_elbo[0],
        'CSVI_RSD': CSVI_elbo[1],
        'SVI_Ind': SVI_elbo[0],
        'SVI': SVI_elbo[1], 
        'CLA': CSL_elbo[1], 
        'Laplace': CSL_elbo[0]
        }

syn_elbo = pd.DataFrame(SYN_ELBO)
df_syn_elbo = syn_elbo.melt(value_vars=['CSVI', 'CSVI_RSD', 'CLA', 'Laplace' ,'SVI_Ind', 'SVI'], 
                        var_name='method', value_name= 'ELBO', ignore_index = True) 
# print(df_syn_elbo)
df_syn_elbo.replace('CSL', 'CLA', inplace=True)

# make plot
f1, ax1 = plt.subplots()
# sns.set_theme(style="whitegrid")
ax1 = sns.violinplot(x = 'method', y = 'ELBO',data = df_syn_elbo, 
                    scale = 'count', inner = 'stick', bw = 0.02, linewidth=0.3, gridsize=1000,
                    order = ['CSVI', 'CSVI_RSD', 'CLA', 'Laplace' ,'SVI_Ind', 'SVI'])
plt.xlabel('')
plt.ylabel('ELBO', fontsize =18)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation  = -15)
ax1.tick_params(axis="both", labelsize= 13)
f1.savefig('figures/sr_syn_elbo.png',bbox_inches='tight',dpi = 500)

# # check laplace
# f1_lap, ax1_lap = plt.subplots()

# ax1_lap = sns.violinplot(x = 'method', y = 'ELBO',data = df_syn_elbo, 
#                     scale = 'count', inner = 'stick', bw = 0.1,
#                     order = ['CSL'])
# plt.xlabel('')
# plt.ylabel('ELBO', fontsize =18)
# # ax1_lap.set_xticklabels(ax1.get_xticklabels(), rotation  = -10)
# ax1_lap.tick_params(axis="both", labelsize= 13)
# f1_lap.savefig('figures/lap_syn_elbo.png',bbox_inches='tight',dpi = 200)


#########################################################
###p2. REAL elbo volin plot
#########################################################


# load results
CSVI_elbo = []
for file in glob.glob(results_dir + "REAL_CSVI_adam*.csv"):
#     print(file)
    df_CSVI = pd.read_csv(file)
    CSVI_elbo.append(np.array(df_CSVI['0']))

SVI_elbo= []
for file in glob.glob(results_dir + "REAL_SVI_adam*.csv"):
        # print(file)
        df_SVI = pd.read_csv(file)
        SVI_elbo.append(np.array(df_SVI['0']))

CSL_elbo= []
for file in glob.glob(results_dir + "REAL_CSL*.csv"):
        df_CSL = pd.read_csv(file)
        CSL_elbo.append(np.array(df_CSL['0']))
# print(SVI_elbo[1][0:60].max())


REAL_ELBO = {'CSVI' : CSVI_elbo[1],
        'CSVI_RSD' : CSVI_elbo[0],
        'SVI_Ind': SVI_elbo[1],
        'SVI': SVI_elbo[0], 
        'CLA': CSL_elbo[0],
        'Laplace': CSL_elbo[1]
}
real_elbo = pd.DataFrame(REAL_ELBO)
df_real_elbo = real_elbo.melt(value_vars=['CSVI', 'CSVI_RSD','CLA', 'Laplace', 'SVI_Ind', 'SVI'], 
                        var_name='method', value_name= 'ELBO', ignore_index = True) 
# print(df_real_elbo)

df_real_elbo.replace('CSL', 'CLA', inplace=True)
 

# make plot
f2, ax2 = plt.subplots()
# sns.set_theme(style="whitegrid")
ax2 = sns.violinplot(x = 'method', y = 'ELBO',data = df_real_elbo, 
                    scale = 'count', inner = 'stick', bw = 0.02, linewidth=0.3, gridsize=1000,
                    order = ['CSVI', 'CSVI_RSD','CLA', 'Laplace', 'SVI_Ind', 'SVI'])
plt.xlabel('')
plt.ylabel('ELBO', fontsize =18)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation  = -15)
ax2.tick_params(axis="both", labelsize= 13)
f2.savefig('figures/sr_real_elbo.png',bbox_inches='tight',dpi = 500)


