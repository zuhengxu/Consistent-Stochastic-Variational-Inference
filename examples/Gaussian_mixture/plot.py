from pandas.io.parsers import read_csv
from scipy.stats import norm
import os, sys,glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats.morestats import _add_axis_labels_title
import seaborn as sns
import pandas as pd
from seaborn import palettes
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from examples.common.synthetic_model import *
from examples.common.results import *
from examples.common.plotting import *





###################
###################
### set up #######
###################
###################

# create figure folder if it does not exist
if not os.path.exists('figures/'):
        os.mkdir('figures/')

#result dir
results_dir = os.path.join(sys.path[0], 'results/VI_results/')


# ## color palette
# col_pal = {"CSVI": '#89bedc', "SVI": 'pink',
#             'CSVI_RSD': 'teal',  'SVI_Ind':'orange',
#             'SVI_SMAP':'plum' ,'SVI_OPT':'olive'}


#########################################################
###synthetic data visualization
#########################################################
# from sklearn.cluster import KMeans
# syn_dat = np.load('../data/syndat_gmm.npy')
# X = syn_dat[:,:2]
# kmeans = KMeans(4, random_state=0)
# labels = kmeans.fit(X).predict(X)
# f,a = plt.subplots()
# plt.scatter(X[:,0], X[:,1],c = labels, s=40, cmap='viridis')
# plt.xticks(fontsize = 15)
# plt.yticks(fontsize = 15)

# f.savefig('figures/gmm_syn_dat.png',bbox_inches='tight',dpi = 500)



#########################################################
###p1. SYN elbo volin plot
#########################################################

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

SYN_ELBO = {'CSVI(adam)' : CSVI_elbo[0],
        'SVI(adam)_Ind': SVI_elbo[0],
        'SVI(adam)_Rand': SVI_elbo[1],
        'CLA': CSL_elbo[0], 
        'Laplace': CSL_elbo[1]
}
syn_elbo = pd.DataFrame(SYN_ELBO)
df_syn_elbo = syn_elbo.melt(value_vars=['CSVI(adam)','CLA', 'Laplace' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'],
                        var_name='method', value_name= 'ELBO', ignore_index = True)
print(df_syn_elbo)

# make plot
f1, ax1 = plt.subplots()
ax1 = sns.violinplot(x = 'method', y = 'ELBO',data = df_syn_elbo[df_syn_elbo['ELBO'] > -1000],
                    scale = 'count', inner = 'stick', bw = 0.1,
                    order = ['CSVI(adam)', 'CLA', 'Laplace', 'SVI(adam)_Ind', 'SVI(adam)_Rand'])
plt.xlabel('')
plt.ylabel('ELBO',fontsize = 18)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 15)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation  = -13)

f1.savefig('figures/gmm_syn_elbo.png',bbox_inches='tight',dpi = 500)

# # check laplace
# f1_lap, ax1_lap = plt.subplots()

# ax1_lap = sns.violinplot(x = 'method', y = 'ELBO',data = df_syn_elbo, 
#                     scale = 'count', inner = 'stick', bw = 0.1,
#                     order = ['CSL'])
# plt.xlabel('')
# plt.ylabel('ELBO', fontsize =18)
# ax1_lap.set_xticklabels(ax1.get_xticklabels(), rotation  = -13)
# ax1_lap.tick_params(axis="both", labelsize= 13)
# # ax1_lap.set_ylim(bottom = -5000)
# f1_lap.savefig('figures/lap_syn_elbo.png',bbox_inches='tight',dpi = 200)




#########################################################
###p2. REAL elbo volin plot
#########################################################

# load results
CSVI_elbo = []
for file in glob.glob(results_dir + "REAL_CSVI_adam*.csv"):
    print(file)
    df_CSVI = pd.read_csv(file)
    CSVI_elbo.append(np.array(df_CSVI['0']))


SVI_elbo= []
for file in glob.glob(results_dir + "REAL_SVI_adam*.csv"):
        print(file)
        df_SVI = pd.read_csv(file)
        SVI_elbo.append(np.array(df_SVI['0']))

CSL_elbo= []
for file in glob.glob(results_dir + "REAL_CSL*.csv"):
        df_CSL = pd.read_csv(file)
        CSL_elbo.append(np.array(df_CSL['0']))

REAL_ELBO = {'CSVI(adam)_Rand' : CSVI_elbo[0],
        'CSVI(adam)_Ind' : CSVI_elbo[1],
        'SVI(adam)_Rand': SVI_elbo[0],
        'SVI(adam)_Ind': SVI_elbo[1], 
        'CLA': CSL_elbo[1], 
        'Laplace': CSL_elbo[0]
}

real_elbo = pd.DataFrame(REAL_ELBO)
df_real_elbo = real_elbo.melt(value_vars=['CSVI(adam)_Ind' , 'CLA',  'Laplace', 'SVI(adam)_Rand', 'SVI(adam)_Ind'],
                        var_name='method', value_name= 'ELBO', ignore_index = True)
print(df_real_elbo)

# make plot
f2, ax2 = plt.subplots()
ax2 = sns.violinplot(x = 'method', y = 'ELBO', data = df_real_elbo,
                    scale = 'count', inner = 'stick', bw = 0.1,
                    order = ['CSVI(adam)_Ind', 'CLA', 'Laplace','SVI(adam)_Ind', 'SVI(adam)_Rand'])
plt.xlabel('')
plt.ylabel('ELBO',fontsize = 18)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 15)
ax2.tick_params(axis="both", labelsize= 13)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation  = -13)

f2.savefig('figures/gmm_real_elbo.png',bbox_inches='tight',dpi = 500)



