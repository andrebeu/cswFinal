import os
import sys
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
from scipy.special import softmax
from itertools import product
import numpy as np
from utils import *
from model import *
from datetime import datetime

## RANDOM GRIDSEARCHING

## timing
startTime = datetime.now() 
tstamp = time.perf_counter_ns() + np.random.randint(999)

## saving dir
GSDIR = 'data/gs1029/'

## param ranges
alpha_min = 0.001
alpha_max = 100
betawi_min = 0.001
betawi_max = 100
betabt_min = 0.001
betabt_max = 100
lmda_min = 0.001
lmda_max = 1.2


## setup
num_seeds = 50
condL = ['blocked','interleaved',
         'early','middle','late'
        ]
print('SETUP','ns=',num_seeds,condL)

idx=0

dfL = []
while True:
    # for idx in range(5):
    print('itr',idx)
    itr_t0 = datetime.now() 
    schargs = {
        'concentration':np.random.uniform(alpha_min,alpha_max),
        'stickiness_wi':np.random.uniform(betawi_min,betawi_max),
        'stickiness_bt':np.random.uniform(betabt_min,betabt_max),
        'sparsity':np.random.uniform(lmda_min,lmda_max),
        'pvar': 0,
        'lrate':1,
        'lratep':1,
        'decay_rate':1,
    }
    semargs = {
        'beta2':False,
        'skipt1':False,
        'ppd_allsch':False
    }
    taskargs = {
        'condition':None,
        'n_train':160,
        'n_test':40
    }
    args = {
        'sem':semargs,
        'sch':schargs,
        'exp':taskargs
    }
    # run
    exp_batch_data = run_batch_exp_curr(num_seeds,args,condL) # [curr,seeds,{data}]
    batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
    mean_acc = batch_acc.mean(1) # mean over seeds
    for ci in range(len(condL)):
        cond_df = pd.DataFrame(data={
            **schargs,
            'cond':condL[ci],
            'trial':range(200),
            'acc':mean_acc[ci]
        })
        dfL.append(cond_df)
    # save
    pd.concat(dfL).to_csv(GSDIR+'df-%i.csv'%tstamp)
    print('ITR TIME TAKEN', datetime.now() - itr_t0)
    ##

    
delta_time = datetime.now() - startTime 
print('DONE')
print('TIME TAKEN',delta_time)
