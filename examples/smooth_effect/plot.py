import os, sys, glob

from scipy.stats.morestats import _add_axis_labels_title
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from autograd.scipy import stats
import autograd.numpy as np
from scipy.stats import norm
from numpy import convolve
import matplotlib.pyplot as plt
from examples.common.smooth_example import *
from examples.common.plotting import *


fig_dir = "figure"
map_dir=  "results"

### 1. unnormalized posterior pdf with SS = (10, 100, 1000, 10000)
f1, axa = plt.subplots(1, 4, figsize=(16,2.8))
c = 0
X_range  = np.linspace(-15, 15, 3001)
for SS in [10, 100, 1000, 10000]:
    # the 23 is just a normalizing constant
    unorm_pdf = lambda x : np.exp(toy_unorm_lpdf(x, SS))
    lY = toy_unorm_lpdf(X_range, SS)
    Y = np.exp(lY)
    Y = Y/Y.max()*100.

    axa[c].set_title('Posterior, N =' +str(SS), size=15)
    axa[c].plot(X_range, Y, color = 'k')

    c+= 1

for ax in axa:
    ax.tick_params(axis="both", labelsize= 15)


f1.savefig(fig_dir +  "/toy_posts.png",bbox_inches='tight',dpi = 400)





### 2. posts with increasing smoothing kernel
f2, axa2 = plt.subplots(1, 4, figsize=(16,2.8))
c = 0
X_range  = np.linspace(-20, 20, 4001)
unorm_pdf = lambda x : np.exp(toy_unorm_lpdf(x, 10))
Y = unorm_pdf(X_range)
axa2[c].set_title('alpha = 0', fontsize = 30)
axa2[c].plot(X_range, Y, color = 'k')

for alpha in [1, 2, 4]:
    c += 1 
    gs_kernel = lambda x: np.exp(-(x**2)/(2*alpha))/np.sqrt(2*np.pi*alpha)
    K = gs_kernel(X_range)
    smth_pdf = convolve(Y, K, 'same')
    smth_pdf = smth_pdf* 100./smth_pdf.max()

    axa2[c].set_title('alpha = ' + str(alpha), fontsize = 30)
    axa2[c].plot(X_range, smth_pdf, color = 'k')
    print(c)

for ax in axa2:
    ax.tick_params(axis="x", labelsize= 20)
    ax.set_yticks([])

f2.savefig(fig_dir +  "/toy_smooth_posts.png", bbox_inches='tight',dpi = 400)





####3. smooth MAP plot with track of 100 trials
f3, axa3 = plt.subplots(1, 4, figsize=(16,2.8))
X_range  = np.linspace(-20, 20, 4001)
c = 0 
K = 100
num_trials = 100


for SS in [10, 100, 1000, 10000]:
    T = np.linspace(X_range.min(),X_range.max(), 401) # change if x_range changes
    bars = [0]*len(T)
    
    unorm_pdf = lambda x : np.exp(toy_unorm_lpdf(x, SS))
    Y = unorm_pdf(X_range)
    alpha= 10*SS**(-0.3)
    gs_kernel = lambda x: np.exp(-(x**2)/(2*alpha))/np.sqrt(2*np.pi*alpha)
    K = gs_kernel(X_range)
    smth_pdf = convolve(Y, K, 'same')
    smth_pdf = smth_pdf/smth_pdf.max()*num_trials

    name = glob.glob(map_dir + '/smth_MAP'+ str(SS)+ '.npy')
    smth_MAP = np.load(name[0])

    s = []
    for i in range(num_trials):
        x_smap = smth_MAP[i]
        bars[int((x_smap+20)/0.1)]+=1 # change if X_range change
        s.append(value(x_smap, smth_pdf))

    axa3[c].set_title('Smoothed MAP, N =' +str(SS), size=15)
    axa3[c].plot(X_range, smth_pdf, color = 'k',alpha= 0.6)
    axa3[c].plot(smth_MAP,s,'ro', markersize=3,color='r',label='Last Iter')
    axa3[c].bar(T, bars, color='b', align = 'center', alpha=0.5, width=0.4, label='Last Iter Count')
    if c == 0: 
        axa3[c].legend(loc='best')
    c+= 1

for ax in axa3:
    ax.tick_params(axis="both", labelsize= 15)

f3.savefig(fig_dir + "/toy_smooth_ct.png",bbox_inches='tight', dpi = 400)
