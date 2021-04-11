from scipy.stats import norm
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# read results
results_dir = os.path.join(sys.path[0], 'results/VI_results/')
df_results = results_concat(results_dir)


## color palette
col_pal = {"CSVI": '#89bedc', "SVI": 'pink',
            'CSVI_RSD': 'teal',  'SVI_Ind':'orange',
            'SVI_SMAP':'plum' ,'SVI_OPT':'olive'}

###################
###################
### create plots###
###################
###################


#########################################################
###p1. line plot to illustrate the results
#########################################################
f1, axa1 = plt.subplots(1, 6, figsize=(16,2.8), sharey = 'row', sharex = 'col' )

# post_patch = mpatches.Patch(color = 'grey', label = 'Posterior')
# csvi_patch = mpatches.Patch(color = '#89bedc', label = 'CSVI')
# csrsd_patch = mpatches.Patch(color = 'turquoise', label = 'CSVI_RSD')
# svi_patch = mpatches.Patch(color = 'pink', label = 'SVI')
# svii_patch  = mpatches.Patch(color = 'orange', label = 'SVI_Ind')
# svis_patch = mpatches.Patch(color = 'plum', label = 'SVI_SMAP')
# svio_patch  = mpatches.Patch(color = 'olive', label = 'SVI_OPT')

np.random.seed(12121212)
c= 0
X_axis = np.linspace(-40 , 40, 1000)
zip_arg = zip(['CSVI', 'CSVI_RSD', 'SVI', 'SVI_Ind','SVI_SMAP' ,'SVI_OPT'],
                ['#89bedc', 'turquoise'  ,'pink','orange', 'plum' ,'olive'],
                ['#0b559f', 'teal' ,'palevioletred','orangered', 'orchid','green'])

for alg, col, curve in zip_arg:
# pick 20 random results and plot the GVB distirbutions
    for i in np.random.choice(range(100), size = 20, replace= False):
        ddf = df_results[df_results.method == alg].reset_index(drop = True)
        mean, sd = ddf.loc[i,'mean_vi'], ddf.loc[i,'sd_vi']

        axa1[c].plot(X_axis, norm.pdf(X_axis, mean, sd), color= curve, alpha = 0.7)
        axa1[c].fill_between(X_axis, norm.pdf(X_axis, mean, sd), color= col, alpha= 0.3)
    axa1[c].plot(X_axis, gaussian_mix(X_axis), color = 'black')
    axa1[c].fill_between(X_axis, gaussian_mix(X_axis), color = 'grey', alpha = 0.5)
    axa1[c].set_title(alg, size = 20)
    c += 1

for ax in axa1:
    ax.tick_params(axis="both", labelsize= 15)

f1.savefig('figures/mixture.png',bbox_inches='tight',dpi = 500)



# #########################################################
# ###p2. elbo volin plot
# #########################################################
f2, ax2 = plt.subplots()
ax2 = sns.violinplot(x = 'method', y = 'elbo',data = df_results,
                    scale = 'count', inner = 'stick', palette= col_pal, bw = 0.15,
                    order = ['CSVI', 'CSVI_RSD', 'SVI', 'SVI_Ind', 'SVI_SMAP', 'SVI_OPT'])
plt.xlabel('')
plt.ylabel('ELBO',fontsize = 18)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 15)


f2.savefig('figures/mixture_elbo.png',bbox_inches='tight',dpi = 500)


# #########################################################
# ###p3. illustrate the smap across different alpha
# #########################################################
f3, ax3 = plt.subplots(1, 5,figsize=(20,2.8))
X_range  = np.linspace(-40, 40, 4001)
df_csvi = pd.read_csv(os.path.join('results/sensitivity/','alpha_elbo_CSVI'+ '.csv' ))
df_csvi_rsd  = pd.read_csv(os.path.join('results/sensitivity/','alpha_elbo_CSVI_RSD'+ '.csv' ))

c = 0
for alpha in [10, 20, 50, 100, 200]:
    T = np.linspace(X_range.min(),X_range.max(), 401) # change if x_range changes
    bars = [0]*len(T)

    Y = gaussian_mix(X_range)
    # scale the plot
    Y = Y/Y.max()*100
    # s = []
    df_alpha = df_csvi[df_csvi.alpha == alpha].reset_index(drop = True)
    for i in range(100):
        x_smap = df_alpha.loc[i, 'mean_initial']
        bars[int((x_smap+40)/0.2)]+=1 # change if X_range change
        # s.append(value(x_smap, Y)))

    ax3[c].set_title('SMAP, alpha =' + str(alpha), size = 20)
    ax3[c].plot(X_range, Y, color = 'k',alpha= 0.6)
    # ax3[c].plot(np.array(df_alpha.mean_initial),s,'ro', markersize=2,color='r',label='Last Iter')
    ax3[c].bar(T, bars, color='red', align = 'center', alpha=0.3, width= 2, label='Last Iter Count')
    print(c)
    if c == 5:
        ax3[c].legend(loc='best')
    c+= 1

for ax in ax3:
    ax.tick_params(axis="both", labelsize= 16)

f3.savefig("figures/alpha_smap.png",bbox_inches='tight', dpi = 500)



# ########################################################################
# ###p4. voilin plots of csvi/csvi_rsd across different alpha
# ########################################################################
df_csvi_add = pd.read_csv(os.path.join('results/sensitivity/','alpha_add_CSVI'+ '.csv' ))
df_csvi_rsd_add  = pd.read_csv(os.path.join('results/sensitivity/','alpha_add_CSVI_RSD'+ '.csv' ))
df_csvi_Add = pd.read_csv(os.path.join('results/sensitivity/','alpha_Add_CSVI'+ '.csv' ))
df_csvi_rsd_Add  = pd.read_csv(os.path.join('results/sensitivity/','alpha_Add_CSVI_RSD'+ '.csv' ))

df_alpha = pd.concat([df_csvi, df_csvi_rsd,df_csvi_add, df_csvi_rsd_add,df_csvi_Add, df_csvi_rsd_Add], 
                    join = "inner", ignore_index= True)
df_alpha= df_alpha.loc[df_alpha['alpha'].isin((20, 50, 200, 2000, 10000, 100000))]
df_alpha_outlier = df_alpha[df_alpha['elbo']< -3.5]
print(df_alpha_outlier)
# df_alpha_outlier = 
f4, ax4 = plt.subplots()
ax4 = sns.violinplot(x = 'alpha', y = 'elbo',data = df_alpha,
                    hue = 'method', split = True ,palette= col_pal,
                    scale = 'count', inner = 'stick',bw = 0.08)
ax4 = sns.stripplot(x = 'alpha', y = 'elbo', data = df_alpha_outlier,
                    hue = 'method', palette= col_pal,  
                    linewidth= 1, jitter=0.01)
plt.xlabel('alpha', fontsize = 18)
plt.ylabel('ELBO', fontsize = 18)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize = 12, ncol = 4)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
f4.savefig('figures/alpha_elbo.png',bbox_inches='tight',dpi = 500)



########################################################################
###p5. quantile line plot of ELBOs across different vi_stepsched
########################################################################


f5, ax5 = plt.subplots()
for alg, color in zip(['CSVI', 'SVI'],['#89bedc', 'pink']):
    df = pd.read_csv(os.path.join('results/sensitivity/','step_'+ alg+ '.csv' ))
    A = np.array([])
    for step in [5,10,15,20,25,30]:
        a  =df[df.vi_stepsched == step].elbo.reset_index(drop= True)
        A = np.append(A, np.array(a))
    Ys= np.transpose(A.reshape(6,100))
    X = np.array([5,10,15,20,25,30])
    lineplot_qtl(X, Ys, n= 1, percentile_min= 25, percentile_max=75, plot_median=True,
                plot_mean=False, color=color, line_color=color, alpha = 0.3, label = alg)
plt.ylim([-10,2])
plt.legend(loc='lower right', fontsize =  15)
plt.ylabel('ELBO', fontsize = 15)
plt.xlabel('learning rate', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize = 12)
f5.savefig('figures/step_elbo1.png',bbox_inches='tight',dpi = 500)






# i = 1
# for alg, color in zip(['CSVI_RSD',  'SVI_Ind','SVI_SMAP' ,'SVI_OPT'],
#                     ['teal', 'orange', 'plum' ,'olive']):
#     df = pd.read_csv(os.path.join('results/sensitivity/','step_'+ alg+ '.csv' ))
#     ff, axx = plt.subplots()
#     A = np.array([])
#     for step in [5,10,15,20,25,30]:
#         a  =df[df.vi_stepsched == step].elbo.reset_index(drop= True)
#         A = np.append(A, np.array(a))
#     Ys= np.transpose(A.reshape(6,100))
#     X  = np.array([5, 19, 15, 20, 30])
#     lineplot_qtl(X, Ys, n=1, percentile_min = 25, percentile_max=75, plot_median=True,
#                     plot_mean=False, color=color, line_color=color, alpha = 0.3,
#                     label = alg)
#     plt.ylim([-10,2])
#     plt.legend(loc='lower right', fontsize =  24)
#     plt.ylabel('ELBO', fontsize = 24)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize = 20)
#     i += 1
#     ff.savefig('figures/step_elbo_'+ alg+ '.png',bbox_inches='tight',dpi = 500)






########################################################################
###p6. SVI, CSVI qunatile of SVI across different step_sched
########################################################################
f6, ax6 = plt.subplots()
step_csvi = pd.read_csv(os.path.join('results/sensitivity/','step_CSVI'+ '.csv' ))
step_svi = pd.read_csv(os.path.join('results/sensitivity/','step_SVI'+  '.csv' ))
step_svi_opt = pd.read_csv(os.path.join('results/sensitivity/','step_SVI_OPT' + '.csv' ))
df_step = pd.concat([step_csvi, step_svi], join = "inner", ignore_index= True)


ax6 = sns.violinplot(x = 'vi_stepsched', y = 'elbo',data = df_step[df_step["elbo"] > -10 ],
                    hue = 'method',split = True, palette= col_pal,
                    scale = 'count', inner = 'stick', bw = 0.1)
# ax6 = sns.stripplot(x = 'vi_stepsched', y = 'elbo', data = df_step,
#                     hue = 'method', palette= col_pal,  
#                     linewidth= 1, jitter=0.01, dodge = True)
plt.ylim([-13,2])
plt.xlabel('')
plt.ylabel('ELBO', fontsize =  18)
plt.legend(loc='lower right', fontsize =  15)
plt.xlabel('learning rate', fontsize = 18)
plt.xticks(fontsize=15)
plt.yticks(fontsize = 15)
f6.savefig('figures/step_elbo_violin.png',bbox_inches='tight',dpi = 500)


########################################################################
###p7. SVI, CSVI qunatile of SVI across different step_sched (scatter)
########################################################################
f7, ax7 = plt.subplots()
ax6 = sns.stripplot(x = 'vi_stepsched', y = 'elbo', data = df_step,
                    hue = 'method', palette= col_pal,  
                    linewidth= 1, jitter=0.01)
# plt.ylim([-13,2])
plt.xlabel('')
plt.ylabel('')
plt.legend(loc='lower right', fontsize =  25)
plt.xlabel('')
plt.xticks(fontsize=25)
plt.yticks(fontsize = 25)
f7.savefig('figures/step_elbo_violin.png',bbox_inches='tight',dpi = 500)

