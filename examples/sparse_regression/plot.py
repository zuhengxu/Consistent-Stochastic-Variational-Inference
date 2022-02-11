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

SYN_ELBO = {'CSVI(adam)_Ind' : CSVI_elbo[0],
        'CSVI(adam)_Rand': CSVI_elbo[1],
        'SVI(adam)_Ind': SVI_elbo[0],
        'SVI(adam)_Rand': SVI_elbo[1], 
        'CSL': CSL_elbo[1], 
        'CSL_Rand': CSL_elbo[0]
        }

syn_elbo = pd.DataFrame(SYN_ELBO)
df_syn_elbo = syn_elbo.melt(value_vars=['CSVI(adam)_Ind', 'CSVI(adam)_Rand', 'CSL', 'CSL_Rand' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'], 
                        var_name='method', value_name= 'ELBO', ignore_index = True) 
print(df_syn_elbo)

# make plot
f1, ax1 = plt.subplots()
# sns.set_theme(style="whitegrid")
ax1 = sns.violinplot(x = 'method', y = 'ELBO',data = df_syn_elbo, 
                    scale = 'count', inner = 'stick', bw = 0.1,
                    order = ['CSVI(adam)_Ind', 'CSVI(adam)_Rand', 'CSL', 'CSL_Rand' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'])
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


REAL_ELBO = {'CSVI(adam)' : CSVI_elbo[1],
        'CSVI(adam)_Rand' : CSVI_elbo[0],
        'SVI(adam)_Ind': SVI_elbo[1],
        'SVI(adam)_Rand': SVI_elbo[0], 
        'CSL': CSL_elbo[0], 
        'CSL_Rand': CSL_elbo[1]
}
real_elbo = pd.DataFrame(REAL_ELBO)
df_real_elbo = real_elbo.melt(value_vars=['CSVI(adam)', 'CSVI(adam)_Rand','CSL', 'CSL_Rand' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'], 
                        var_name='method', value_name= 'ELBO', ignore_index = True) 
# print(df_real_elbo)


# make plot
f2, ax2 = plt.subplots()
# sns.set_theme(style="whitegrid")
ax2 = sns.violinplot(x = 'method', y = 'ELBO',data = df_real_elbo, 
                    scale = 'count', inner = 'stick', bw = 0.1,
                    order = ['CSVI(adam)', 'CSVI(adam)_Rand' , 'CSL', 'CSL_Rand' ,'SVI(adam)_Ind', 'SVI(adam)_Rand'])
plt.xlabel('')
plt.ylabel('ELBO', fontsize =18)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation  = -15)
ax2.tick_params(axis="both", labelsize= 13)
f2.savefig('figures/sr_real_elbo.png',bbox_inches='tight',dpi = 500)




# #########################################################
# ###p3. Posterior visualization and local optima (syn)
# #########################################################

# # read stan trace to get KDE
# stan_result_path = 'results/stan/'
# trace = pickle.load(open(os.path.join(stan_result_path, 'sr_trace.pkl'), 'rb'))

# f3, ax3 = plt.subplots()
# beta0_trace = pd.DataFrame({'post' : trace['beta'][:,0]})
# sns.kdeplot(beta0_trace['post'], color = 'tab:blue', linewidth = 2, shade = True, label = 'posterior(coord = 1)')

# from scipy.stats import norm
# x = np.linspace(-10,20,1000)
# plt.plot(x, norm.pdf(x, 5.58, 1.72), label = 'ELBO = -0.8', ls = '--', color = "tab:red",linewidth = 3)
# plt.plot(x, norm.pdf(x, 0.01, 0.1), label = 'ELBO = -5.92',ls = '--', color = "tab:orange" , linewidth = 3)
# plt.legend(fontsize = 12)
# plt.ylim( 0, 0.5)
# plt.xlabel('')
# plt.ylabel('Density', fontsize= 18)
# f3.savefig('figures/sr_post_visual.png',bbox_inches='tight',dpi = 500)




# ####################################################################
# ##p4. postyerior pairwise contour plot (syn)
# ####################################################################
# #read stan trace to get KDE
# stan_result_path = 'results/stan/'
# trace = pickle.load(open(os.path.join(stan_result_path, 'sr_trace.pkl'), 'rb'))


# if not os.path.exists( os.path.join(stan_result_path, 'fig.pkl')):
#     df_beta = pd.DataFrame(trace['beta'])
#     df_beta.columns = ['beta'+ str(i+1) for i in range(5)]
#     dff = df_beta[df_beta.index % 20 == 0]
#     f4 = sns.PairGrid(dff)
#     f4.map_lower(sns.kdeplot, fill=True)
#     f4.map_diag(sns.histplot, kde=True)

#     with open(os.path.join(stan_result_path, 'fig.pkl'), 'wb') as f:
#         pickle.dump(f4,f)

# else:
#     f4 = pickle.load(open(os.path.join(stan_result_path, 'fig.pkl'), 'rb'))

# f4.savefig('figures/sr_syn_pairwise_countour.png',bbox_inches='tight',dpi = 500)






# #####################################################################
# ###p5. beta12; beta23 countour plot: Posterior visualization and local optima (syn) 
# #####################################################################
# # read stan trace to get KDE
# stan_result_path = 'results/stan/'
# trace = pickle.load(open(os.path.join(stan_result_path, 'sr_trace.pkl'), 'rb'))
# df_beta = pd.DataFrame(trace['beta'])
# df_beta.columns = ['beta'+ str(i+1) for i in range(5)]
# # thining MCMC samples every 10
# dff = df_beta[df_beta.index % 10 == 0]

# # read VI results
# df_cs = pd.read_csv('results/VI_results/SYN_CSVI_adam_SMAP_adamInd.csv')
# df_csrand = pd.read_csv('results/VI_results/SYN_CSVI_adam_SMAP_adamRandom.csv')
# df_srand = pd.read_csv('results/VI_results/SYN_SVI_adam_PriorRandom.csv')
# df_lap = pd.read_csv('results/VI_results/SYN_CSL_SMAP_adamRandom.csv')

# f6, ax6 = plt.subplots(1,2,figsize=(10,2.8))
# sns.kdeplot(data=dff, x="beta1", y="beta2", fill = True, color = "grey", bw_adjust= 0.5, ax= ax6[0])
# sns.kdeplot(data=dff, x="beta3", y="beta2", fill = True, color = "grey", bw_adjust= 0.5, ax = ax6[1])
# for c in range(2):

#     for i,color,l in zip([0, 2], ['orange', 'purple'], ['solid', 'dashed']):
#         elbo = df_cs.iloc[i][0]
#         _,mu,L = de_flatten(df_cs.iloc[i][1:])
#         if c == 0:
#             x, y = npp.mgrid[-5:15:.01, -2:2:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
#         else:
#             x, y = npp.mgrid[-6:6:.01, -1:1:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2][::-1], np.fliplr(np.flipud(np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)))
#         rv = mvnorm(u, Sigma)
#         ax6[c].contour(x, y, rv.pdf(pos), levels = [0.001, 0.70] , colors = color, alpha = .7, 
#                         linestyles = l, linewidths = 2.)

    
#     for i, color,l in zip([12,29], ['green', 'black'], ['solid', 'dashdot']):
#         elbo = df_srand.iloc[i][0]
#         _,mu,L = de_flatten(df_srand.iloc[i][1:])
#         if c == 0:
#             x, y = npp.mgrid[-5:15:.01, -2:2:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
#         else:
#             x, y = npp.mgrid[-6:6:.01, -1:1:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2][::-1], np.fliplr(np.flipud(np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)))
#         rv = mvnorm(u, Sigma) 
#         ax6[c].contour(x, y, rv.pdf(pos),levels = [0.03, 1.], colors = color, alpha = 0.7, 
#                         linestyles = l, linewidths = 2.)

# ax6[0].set_xlim(xmin = -2. , xmax  =13)
# ax6[0].set_ylim(ymin = -2., ymax = 2)
# ax6[1].set_xlim(xmin = -5.5 , xmax  =5.5)
# ax6[1].set_ylim(ymin = -2., ymax =  2)
# # generate legend 
# gl, = plt.plot([], color = 'orange',linestyle= 'solid')
# rl, = plt.plot([], color = 'purple', linestyle = 'dashed')
# ol, = plt.plot([], color = 'green', linestyle = 'solid')
# bl, = plt.plot([], color = 'black', linestyle = 'dashdot')
# plt.legend([gl, rl, ol, bl],['ELBO = ' + str(elbo) for elbo in [-1, -2, -6, -7]], 
#             bbox_to_anchor=(1.35, 1), borderaxespad=0.)
# for ax in ax6:
#     ax.tick_params(axis="both", labelsize= 15)
# f6.savefig('figures/sr_syn_vis.png',bbox_inches='tight',dpi = 500)



# ###########3
# #### slice plot
# #######################
# from examples.common.sparse_reg_model import syn_lpdf
# df_init = pd.read_csv('results/initials/SYN_SMAP_adamInd.csv')
# res_csl = np.array(df_lap)[0, 1:]
# init = np.array(df_init)[0]

# d, mu_csl, L_csl = de_flatten(res_csl)
# d, smap, L_init = de_flatten(init) 



# f_lap, ax_lap = plt.subplots()
# def lpdf_slice(x):
#     a = np.concatenate((np.array([x]), smap[1:]))
#     return -syn_lpdf(a)

# x = np.linspace(-20, 20, 1000)
# ax_lap.plot(x, [lpdf_slice(a) for a in x])
# # f_lap.savefig('figures/slice.png',bbox_inches='tight',dpi = 500)