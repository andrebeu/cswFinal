#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob as glob
import pandas as pd
import numpy as np


# In[2]:


fL = glob('data/gs1021/rawacc/acc-*')
condL = ['blocked','interleaved','early','middle','late']
accL = []
for f in fL:
  ## load single param at a time 
  accseeds = np.load(f) # cond,seed,trials
  acc = accseeds.mean(1)
  paramD = {}
  param_names = ['alpha','betawi','betabt','lmda']
  param_vals = [i.split('_')[-1] for i in f.split('/')[-1].split('-')[1:]][:4]
  paramD = {**paramD,**dict(zip(param_names,param_vals))}
  ## make dataframe for single param
  for ci in range(len(condL)):
    accdf = pd.DataFrame(data={
      **paramD,
      'cond':condL[ci],
      'trial':np.arange(200),
      'acc':acc[ci],
    })
    accL.append(accdf)

gsdf = pd.concat(accL)


# In[4]:


gsdf.to_csv('data/gs1021/gsdf.csv')

