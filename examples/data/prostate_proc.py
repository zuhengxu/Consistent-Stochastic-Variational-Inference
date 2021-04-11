from os import uname
import pandas as pd
import io
import requests
import autograd.numpy as np
#read prostate data from url
url="http://www.web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data"
s=requests.get(url).content
prostate_dat=pd.read_csv(io.StringIO(s.decode('utf-8')), sep="\t")
#drop the row index , drop train (boolean)
prostate_dat = prostate_dat.drop(columns = ['Unnamed: 0', 'train'])
# put response lpsa at first col
colnames  = prostate_dat.columns.values
prostate_dat =prostate_dat[np.hstack((colnames[-1], colnames[:-1])) ]
prostate_dat =  np.array(prostate_dat)
# subsample dataset to n = 30
np.random.seed(2021)
Dat = prostate_dat[np.random.choice(prostate_dat.shape[0], 30, replace = False),:]

# standarize the covariates
X_raw =  Dat[:, 1:]
m = np.mean(X_raw, axis =0)
sd = np.std(X_raw, axis = 0)
X = (X_raw - m)/sd
Y = Dat[:, 0]; Y = (Y-np.mean(Y))/np.std(Y)
Dat_proc = np.hstack((Y[:, None], X))
np.save('prostate.npy', Dat_proc)



