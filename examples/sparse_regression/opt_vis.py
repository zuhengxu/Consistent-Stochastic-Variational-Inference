from pandas.io.parsers import read_csv
from scipy.stats import norm
import os, sys,glob,pickle
import matplotlib.pyplot as plt
import numpy as npp
from scipy.stats import multivariate_normal as mvnorm
import seaborn as sns
import pandas as pd
from seaborn import palettes
from examples.common.elbo import multi_ELBO
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from examples.common.results import *
from examples.common.plotting import *
from VI.util import de_flatten




#create figure folder if it does not exist
if not os.path.exists('figures/'):
        os.mkdir('figures/')

#result dir
results_dir = os.path.join(sys.path[0], 'results/VI_results/')


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
# f_lap.savefig('figures/slice.png',bbox_inches='tight',dpi = 500)


















########################
# REAL data 
#############################
# read stan trace to get KDE
stan_result_path = 'results/stan/'
# stan_result_path = '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan'
trace = pickle.load(open(os.path.join(stan_result_path, 'sr_real_trace.pkl'), 'rb'))
# trace = pickle.load(open('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan/sr_real_trace.pkl', 'rb'))

####################################################################
##p4. postyerior pairwise contour plot (REAL)
####################################################################
#read stan trace to get KDE

if not os.path.exists( os.path.join(stan_result_path, 'real_fig.pkl')):
    df_beta = pd.DataFrame(trace['beta'])
    df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
    dff = df_beta[df_beta.index % 10 == 0]
    f4 = sns.PairGrid(dff)
    f4.map_lower(sns.kdeplot, fill=True, bw_method = 1)
    f4.map_diag(sns.kdeplot, bw_method = 1)

    with open(os.path.join(stan_result_path, 'real_fig.pkl'), 'wb') as f:
        pickle.dump(f4,f)

else:
    f4 = pickle.load(open(os.path.join(stan_result_path, 'real_fig.pkl'), 'rb'))

f4.savefig('figures/sr_realsmooth_pairwise_countour.png',bbox_inches='tight',dpi = 500)






# #####################################################################
# ###p5. beta12; beta23 countour plot: Posterior visualization and local optima (syn) 
# #####################################################################
# # read stan trace to get KDE
# df_beta = pd.DataFrame(trace['beta'])
# df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
# # thining MCMC samples every 10
# dff = df_beta[df_beta.index % 10 == 0]

# # read VI results
df_cs = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamInd.csv')
df_csrand = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamRandom.csv')
df_srand = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorRandom.csv')
df_lap = pd.read_csv('results/VI_results/REAL_CSL_PriorRandom.csv')
df_csl = pd.read_csv('results/VI_results/REAL_CSL_SMAP_adamRandom.csv')


# pick candidates
a = df_lap.iloc[3]
b = df_cs.iloc[1]
c = df_srand.iloc[3]
d = df_csl.iloc[95]
e = df_srand.iloc[30]

# _, m0, L0 = de_flatten(a[1:]) #-6
# _, m1, L1 = de_flatten(b[1:]) # -13
# _, m2, L2 = de_flatten(c[1:]) # -21 
# _, m3, L3 = de_flatten(d[1:]) # -23  
# _, m4, L4 = de_flatten(e[1:]) # -11



# f6, ax6 = plt.subplots(1,2,figsize=(10,2.8))
# sns.kdeplot(data=dff, x="beta1", y="beta2", fill = True, color = "grey", bw_adjust= 0.2, ax= ax6[0])
# sns.kdeplot(data=dff, x="beta2", y="beta3", fill = True, color = "grey", bw_adjust= 0.2, ax = ax6[1])

# for c in range(2):
#     for color,l, mu, L in zip(['green', 'blue', 'purple', 'orange'], ['solid', 'dashed','solid', 'dashdot'],
#                                 [ m1, m2, m0, m4], [ L1, L2, L0, L4,]):
#         if c == 0:
#             x, y = npp.mgrid[-1:2:.01, -1:1:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
#         else:
#             x, y = npp.mgrid[-1:1:.01, -1.5:1:.01]
#             pos = npp.dstack((x, y))
#             u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
#             # u, Sigma = mu[c:c+2][::-1], np.fliplr(np.flipud(np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)))
#         rv = mvnorm(u, Sigma)
#         sd = npp.sqrt(npp.diagonal(Sigma))
#         lev1 = rv.pdf(u + sd)
#         lev2 = rv.pdf(u + 2*sd)

#         ax6[c].contour(x, y, rv.pdf(pos), levels = [lev2, lev1] , colors = color, alpha = .5, 
#                         linestyles = l, linewidths = 2.)

# ax6[0].set_xlim(xmin = -1. , xmax  = 2)
# ax6[0].set_ylim(ymin = -1., ymax = 1)
# ax6[1].set_xlim(xmin = -1 , xmax  = 1)
# ax6[1].set_ylim(ymin = -1.5, ymax =  1)
# # generate legend 
# gl, = plt.plot([], color = 'purple',linestyle= 'solid')
# rl, = plt.plot([], color = 'orange', linestyle = 'dashed')
# ol, = plt.plot([], color = 'green', linestyle = 'solid')
# bl, = plt.plot([], color = 'blue', linestyle = 'dashdot')
# plt.legend([gl, rl, ol, bl],['ELBO = ' + str(elbo) for elbo in [-6, -11, -13, -21]], 
#             bbox_to_anchor=(1.35, 1), borderaxespad=0.)
# for ax in ax6:
#     ax.tick_params(axis="both", labelsize= 13)
# f6.savefig('figures/sr_real_vis.png',bbox_inches='tight',dpi = 500)



#####################################################################
###p5. beta12; beta23 countour plot: Posterior visualization and local optima (syn) 
#####################################################################

os.chdir('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression')
# read stan trace to get KDE
stan_result_path = 'results/stan/'
# stan_result_path = '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan'
trace = pickle.load(open(os.path.join(stan_result_path, 'sr_real_trace.pkl'), 'rb'))
# read stan trace to get KDE
df_beta = pd.DataFrame(trace['beta'])
df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
# thining MCMC samples every 10
dff = df_beta[df_beta.index % 10 == 0]

# read VI results
df_cs = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamInd.csv')    # 6, 14
df_csrand = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamRandom.csv')
df_svi = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorRandom.csv')
df_svi_ind = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorInd.csv')
df_lap = pd.read_csv('results/VI_results/REAL_CSL_PriorRandom.csv')
df_csl = pd.read_csv('results/VI_results/REAL_CSL_SMAP_adamInd.csv')

# x = np.array(a[1:])
# multi_ELBO(real_lpdf, x)


def get_vis(df, ids, name):
    # pickle the ksd 
    if not os.path.exists( os.path.join('figures', 'real_ksd.pkl')):
        fig_obj = plt.subplots(2,2,figsize=(10,7))
        sns.kdeplot(data=dff, x="beta1", y="beta2", fill = True, color = "grey", bw_adjust= 0.2, ax = fig_obj[1][0, 0])
        sns.kdeplot(data=dff, x="beta2", y="beta3", fill = True, color = "grey", bw_adjust= 0.2, ax = fig_obj[1][0, 1])
        sns.kdeplot(data=dff, x="beta3", y="beta4", fill = True, color = "grey", bw_adjust= 0.2, ax = fig_obj[1][1, 0])
        sns.kdeplot(data=dff, x="beta4", y="beta5", fill = True, color = "grey", bw_adjust= 0.2, ax = fig_obj[1][1, 1])

        with open(os.path.join('figures', 'real_ksd.pkl'), 'wb') as f:
            pickle.dump(fig_obj,f)
        f8, ax8 =  pickle.load(open(os.path.join('figures/', 'real_ksd.pkl'), 'rb'))
    else:
        f8, ax8 =  pickle.load(open(os.path.join('figures/', 'real_ksd.pkl'), 'rb'))

    elbo = []
    for c in range(4):
        for id,color,l in zip(ids, 
                            ['purple', 'orange', 'green', 'blue'], 
                            ['solid', 'dashed','solid', 'dashdot']):
            dat = df.iloc[id]
            elbo.append(npp.round(dat[0],2))
            _, mu, L= de_flatten(dat[1:]) 
            x, y = npp.mgrid[-1:2:.01, -1.5:1:.01]
            pos = npp.dstack((x, y))
            u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
            rv = mvnorm(u, Sigma)
            sd = npp.sqrt(npp.diagonal(Sigma))
            lev1 = rv.pdf(u + sd)
            lev2 = rv.pdf(u + 2*sd)

            ax8[int(npp.floor(c/2)), c%2].contour(x, y, rv.pdf(pos), levels = [lev2, lev1] , colors = color, alpha = .5, 
                            linestyles = l, linewidths = 2.)

    ax8[0, 0].set_xlim(xmin = -1. , xmax  = 2)
    ax8[0, 0].set_ylim(ymin = -1., ymax = 1)
    ax8[0, 1].set_xlim(xmin = -1 , xmax  = 1)
    ax8[0, 1].set_ylim(ymin = -1, ymax =  1)
    ax8[1, 0].set_xlim(xmin = -1 , xmax  = 1)
    ax8[1, 0].set_ylim(ymin = -1, ymax =  1)
    ax8[1, 1].set_xlim(xmin = -1 , xmax  = 1)
    ax8[1, 1].set_ylim(ymin = -1, ymax =  1)
    # generate legend 
    gl, = plt.plot([], color = 'purple',linestyle= 'solid')
    rl, = plt.plot([], color = 'orange', linestyle = 'dashed')
    ol, = plt.plot([], color = 'green', linestyle = 'solid')
    bl, = plt.plot([], color = 'blue', linestyle = 'dashdot')
    plt.legend([gl, rl, ol, bl],['ELBO = ' + str(e) for e in elbo], 
                bbox_to_anchor=(1.35, 1), borderaxespad=0.)
    for (ax, aax) in ax8:
        ax.tick_params(axis="both", labelsize= 13)
        aax.tick_params(axis="both", labelsize= 13)
    f8.savefig(os.path.join('figures/', name),bbox_inches='tight',dpi = 500)


get_vis(df_cs, [0,6, 9, 10], 'vis_csvi')
get_vis(df_csrand, [0, 1, 3, 20], 'vis_csrand')
get_vis(df_svi, [27, 1, 2, 51], 'vis_csrand')
get_vis(df_csl, [7, 99, 1, 14], 'vis_csl')
get_vis(df_lap, [54,0, 5, 17], 'vis_lap1')
get_vis(df_lap, [19, 40, 43, 57], 'vis_lap2')
