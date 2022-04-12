#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from scipy.special import softmax
from itertools import product

import numpy as np
from utils import *
from model import *
import time 
import sys

##
tstamp = time.perf_counter_ns()
gsdir = 'gs0828'
##

# c,stwi,stbt,sp,pvar,lrate,lratep,decay
param_strin = str(sys.argv[1])
c,stwi,stbt,sp,pvar,lr,lrp,dy = param_strin.split()

## default params
expargs = {
  'condition':'blocked',
  'n_train':160,
  'n_test':40
}

schargs = {
    'concentration':float(c),
    'stickiness_wi':float(stwi),
    'stickiness_bt':float(stbt),
    'sparsity':float(sp),
    'pvar': float(pvar),
    'lrate':float(lr),
    'lratep':float(lrp),
    'decay_rate':float(dy),
} 
semargs = {
  'beta2':False
}
args = {
    'sem':semargs,
    'sch':schargs,
    'exp':expargs
}
param_str = "-".join(["%s_%.3f"%(i,j) for i,j in schargs.items()])
param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in semargs.items()])
print('params',param_str)

# sweep over concentration
p_name = 'stickiness_wi'
p_vals = np.arange(600,701,10)

ns = 50
dfL = []
condL = ['blocked','interleaved','early','middle','late']
for idx,p_val in enumerate(p_vals):
  print(idx/len(p_vals))
  
  args['sch'][p_name] = p_val  
  exp_batch_data = run_batch_exp_curr(ns,args,condL)
  ## acc
  batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
  mean_acc = batch_acc.mean(1)
  test_acc = mean_acc[:,-40:].mean(1) # curr  
  
  ## plt each param
  # plt.figure(figsize=(20,10))
  # for idx in range(len(condL)):
  #   plt.plot(mean_acc[idx],label=condL[idx])
  # plt.legend()
  # param_str = "-".join(["%s_%.3f"%(i,j) for i,j in schargs.items()])
  # param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in semargs.items()])
  # plt.savefig('data/%s/figures/gsdf%i-%s.png'%(gsdir,tstamp,param_str))

  ## record
  gsD = {
    **schargs,
    **dict(zip(condL,test_acc))
  }
  dfL.append(gsD)
  

gsdf = pd.DataFrame(dfL)

gsdf.to_csv('data/%s/gsdf%i-%s.csv'%(gsdir,tstamp,param_str))

print('done')
