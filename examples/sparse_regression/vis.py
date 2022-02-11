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




#create figure folder if it does not exist
if not os.path.exists('figures/'):
        os.mkdir('figures/')

#result dir
results_dir = os.path.join(sys.path[0], 'results/VI_results/')


# read stan trace to get KDE
# stan_result_path = 'results/stan/'
stan_result_path = '/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan'
# trace = pickle.load(open(os.path.join(stan_result_path, 'sr_real_trace.pkl'), 'rb'))
trace = pickle.load(open('/home/zuheng/Research/Consistent-Stochastic-Variational-Inference/examples/sparse_regression/results/stan/sr_real_trace.pkl', 'rb'))

# #########################################################
# ###p3. Posterior visualization and local optima (real)
# #########################################################

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




####################################################################
##p4. postyerior pairwise contour plot (syn)
####################################################################
#read stan trace to get KDE

if not os.path.exists( os.path.join(stan_result_path, 'real_fig.pkl')):
    df_beta = pd.DataFrame(trace['beta'])
    df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
    dff = df_beta[df_beta.index % 20 == 0]
    f4 = sns.PairGrid(dff)
    f4.map_lower(sns.kdeplot, fill=True)
    f4.map_diag(sns.histplot, kde=True)

    with open(os.path.join(stan_result_path, 'real_fig.pkl'), 'wb') as f:
        pickle.dump(f4,f)

else:
    f4 = pickle.load(open(os.path.join(stan_result_path, 'real_fig.pkl'), 'rb'))

f4.savefig('figures/sr_real_pairwise_countour.png',bbox_inches='tight',dpi = 500)






#####################################################################
###p5. beta12; beta23 countour plot: Posterior visualization and local optima (syn) 
#####################################################################
# read stan trace to get KDE
df_beta = pd.DataFrame(trace['beta'])
df_beta.columns = ['beta'+ str(i+1) for i in range(8)]
# thining MCMC samples every 10
dff = df_beta[df_beta.index % 10 == 0]

# read VI results
df_cs = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamInd.csv')
df_csrand = pd.read_csv('results/VI_results/REAL_CSVI_adam_SMAP_adamRandom.csv')
df_srand = pd.read_csv('results/VI_results/REAL_SVI_adam_PriorRandom.csv')
df_lap = pd.read_csv('results/VI_results/REAL_CSL_PriorRandom.csv')
df_lap_bad = pd.read_csv('results/VI_results/REAL_CSL_SMAP_adamRandom.csv')


# pick candidates
a = df_lap.iloc[3]
b = df_cs.iloc[1]
c = df_srand.iloc[3]
d = df_lap_bad.iloc[95]
e = df_srand.iloc[30]

_, m0, L0 = de_flatten(a[1:]) #-6
_, m1, L1 = de_flatten(b[1:]) # -13
_, m2, L2 = de_flatten(c[1:]) # -21 
_, m3, L3 = de_flatten(d[1:]) # -23  
_, m4, L4 = de_flatten(e[1:]) # -11



f6, ax6 = plt.subplots(1,2,figsize=(10,2.8))
sns.kdeplot(data=dff, x="beta1", y="beta2", fill = True, color = "grey", bw_adjust= 0.5, ax= ax6[0])
sns.kdeplot(data=dff, x="beta2", y="beta3", fill = True, color = "grey", bw_adjust= 0.5, ax = ax6[1])

for c in range(2):
    for color,l, mu, L in zip(['green', 'blue', 'purple', 'orange'], ['solid', 'dashed','solid', 'dashdot'],
                                [ m1, m2, m0, m4], [ L1, L2, L0, L4,]):
        if c == 0:
            x, y = npp.mgrid[-1:2:.01, -1:1:.01]
            pos = npp.dstack((x, y))
            u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
        else:
            x, y = npp.mgrid[-1:1:.01, -1.5:1:.01]
            pos = npp.dstack((x, y))
            u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
            # u, Sigma = mu[c:c+2][::-1], np.fliplr(np.flipud(np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)))
        rv = mvnorm(u, Sigma)
        sd = npp.sqrt(npp.diagonal(Sigma))
        lev1 = rv.pdf(u + sd)
        lev2 = rv.pdf(u + 2*sd)

        ax6[c].contour(x, y, rv.pdf(pos), levels = [lev2, lev1] , colors = color, alpha = .5, 
                        linestyles = l, linewidths = 2.)

    
    # for i, color,l in zip([12,29], ['green', 'black'], ['solid', 'dashdot']):
    #     elbo = df_srand.iloc[i][0]
    #     _,mu,L = de_flatten(df_srand.iloc[i][1:])
    #     if c == 0:
    #         x, y = npp.mgrid[-5:15:.01, -2:2:.01]
    #         pos = npp.dstack((x, y))
    #         u, Sigma = mu[c:c+2], np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)
    #     else:
    #         x, y = npp.mgrid[-6:6:.01, -1:1:.01]
    #         pos = npp.dstack((x, y))
    #         u, Sigma = mu[c:c+2][::-1], np.fliplr(np.flipud(np.dot(L[c:c+2, c:c+2], L[c:c+2, c:c+2].T)))
    #     rv = mvnorm(u, Sigma) 
    #     ax6[c].contour(x, y, rv.pdf(pos),levels = [0.03, 1.], colors = color, alpha = 0.7, 
    #                     linestyles = l, linewidths = 2.)

ax6[0].set_xlim(xmin = -1. , xmax  = 2)
ax6[0].set_ylim(ymin = -1., ymax = 1)
ax6[1].set_xlim(xmin = -1 , xmax  = 1)
ax6[1].set_ylim(ymin = -1.5, ymax =  1)
# generate legend 
gl, = plt.plot([], color = 'purple',linestyle= 'solid')
rl, = plt.plot([], color = 'orange', linestyle = 'dashed')
ol, = plt.plot([], color = 'green', linestyle = 'solid')
bl, = plt.plot([], color = 'blue', linestyle = 'dashdot')
plt.legend([gl, rl, ol, bl],['ELBO = ' + str(elbo) for elbo in [-6, -11, -13, -21]], 
            bbox_to_anchor=(1.35, 1), borderaxespad=0.)
for ax in ax6:
    ax.tick_params(axis="both", labelsize= 13)
f6.savefig('figures/sr_real_vis.png',bbox_inches='tight',dpi = 500)
