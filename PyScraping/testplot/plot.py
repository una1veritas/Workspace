'''
Created on 2018/01/31

@author: sin
'''
%matplotlib inline
import numpy as np
import pandas as pd

idx = pd.date_range('2016/06/01', '2016/07/31 23:59', freq='T')
dn = np.random.randint(2, size=len(idx))*2-1
rnd_walk = np.cumprod(np.exp(dn*0.0002))*100
