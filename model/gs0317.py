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
# setting
GSDIR = 'data/gs0317/'
DEBUG = False

## param ranges
alpha_min = 0.001
alpha_max = 100
betawi_min = 0.001
betawi_max = 100
betabt_min = 0.001
betabt_max = 100
lmda_min = 0.001
lmda_max = 1.2
lratep_min = 1
lratep_max = 1
decay_min = 1
decay_max = 1


## setup
num_seeds = 50
condL = ['blocked','interleaved',
         'early','middle','late'
        ]
print('SETUP','ns=',num_seeds,condL)

## timing
startTime = datetime.now() 
tstamp = time.perf_counter_ns() + np.random.randint(999)

def recordable(acc):
    """ acc [nconds,trials] 
    conds: B,I,E,M,L
    """
    if DEBUG:
        return True
    ## Btest>85
    Btest = acc[0,-40:].mean() > 0.85
    ## Iblock2>50
    Ib2 = acc[1,80:120].mean() >= 0.5
    return Btest&Ib2


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
        'lratep':np.random.uniform(lratep_min,lratep_max),
        'decay_rate':np.random.uniform(decay_min,decay_max),
    }
    #~~~#
    schargs['stickiness_wi'] = schargs['stickiness_bt']
    #~~~#
    semargs = {
        'beta2':False,
        'skipt1':True,
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
    batch_acc = unpack_acc(exp_batch_data,mean_over_tsteps=False) # curr,seeds,tsteps,trials
    mean_acc = batch_acc.mean(1) # mean over seeds
    if recordable(mean_acc.mean(1)): # mean over tsteps
        ## record
        # build df 
        for ci in range(len(condL)):
            cond_df = pd.DataFrame(data={
                **schargs,
                **semargs,
                'cond':condL[ci],
                'trial':range(200),
                'acc2':mean_acc[ci,0],
                'acc3':mean_acc[ci,1]
            })
            dfL.append(cond_df)
        # update csv
        pd.concat(dfL).to_csv(GSDIR+'df-%i.csv'%tstamp)
        print('## recorded')
    else:
        print('** not recorded')
    print('ITR TIME TAKEN', datetime.now() - itr_t0)
    ##

    
delta_time = datetime.now() - startTime 
print('DONE')
print('TIME TAKEN',delta_time)
