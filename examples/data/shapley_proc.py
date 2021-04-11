from os import uname
import pandas as pd
import io
import requests
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
#read prostate data from url

df = pd.read_csv('/home/zuheng/Research/Asymptotic_Optimization_Properties_for_VI/code/examples/data/Shapley_galaxy_dat.txt', 
                delim_whitespace=True).dropna()

df = df[df['V'] < 25000]
conv = (np.pi/180.)
r = df['V']
RA = df['R.A.']*conv
DEC   = df['Dec.']*conv
dRA = RA.max()-RA.min()
dDEC = DEC.max()-DEC.min()
yRA = (RA-(RA.min()+dRA*0.5))*(np.pi/180.)
yDEC  = (DEC-(DEC.min()+dDEC*0.5))*(np.pi/180.)
rRA =r*np.sin(yRA)
rDEC=r*np.sin(yDEC)
X = np.vstack((rRA,r))
X=X.T
Xs = StandardScaler().fit_transform(X)

# subsample to 500 datapoint
idx = np.random.randint(Xs.shape[0], size = 500)
Xs = Xs[idx, :]
np.save('shapley_proc.npy', Xs)

