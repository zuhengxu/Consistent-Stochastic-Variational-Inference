import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import autograd.numpy as np 
from scipy.stats import norm
import os
from examples.common.smooth_example import *
from VI.MAP import smooth_1d_MAP



# directory 
base_path = "/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/experiments/GVB"
directory = 'map_save'

SS  = int(sys.argv[1])

smth_map_lrt = lambda itr: 15/(1+ itr*(0.9))
num_trials = 100

sx = []
alpha= 10*SS**(-0.3)
unorm_pdf = lambda x : np.exp(toy_unorm_lpdf(x, SS))

for trials in range(num_trials):
    x0 = np.array([(np.random.random() - 0.5)*30])
    print("ini = ", x0)
    x_opt = smooth_1d_MAP(x0, unorm_pdf, smth_map_lrt, alpha)
    sx.append(x_opt[0])
    print(sx)


file_name = os.path.join( base_path, directory, "smth_MAP" + str(SS))
mm = np.array(sx)
np.save(file_name, mm)

