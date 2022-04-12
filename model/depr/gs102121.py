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

## timing
startTime = datetime.now() 

## saving dir
GSDIR = 'data/gs1021/'

## param input
param_strin = str(sys.argv[1])
c,stwi,stbt = param_strin.split()
alfa = float(c)
bwi = float(stwi)
bbt = float(stbt)

schargs = {
    'concentration':alfa,
    'stickiness_wi':bwi,
    'stickiness_bt':bbt,
    'sparsity':00,
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


## setup

num_seeds = 25
condL = ['blocked','interleaved',
         'early','middle','late'
        ]
print('SETUP','ns=',num_seeds,condL)

## internal sweep
p_name = 'sparsity'
sparsityL = list(np.arange(0.01,2,0.1)) + list(np.arange(0.01,0.3,0.025))
p_vals = sparsityL
print('spar list',sparsityL)

## runtime
for idx,p_val in enumerate(p_vals):
    print(idx/len(p_vals))
    # run
    args['sch'][p_name] = p_val  
    exp_batch_data = run_batch_exp_curr(num_seeds,args,condL) # [curr,seeds,{data}]
    batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
    # save
    param_str = "-".join(["%s_%.3f"%(i,j) for i,j in args['sch'].items()])
    param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in args['sem'].items()])
    np.save(GSDIR+'rawacc/acc-'+param_str,batch_acc)

delta_time = datetime.now() - startTime 
print('DONE')
print('TIME TAKEN',delta_time)

# def calc_adjrand(exp_batch_data):
#   arscores = -np.ones([len(condL),ns,3])
#   for curr_idx in range(len(condL)):
#     for seed_idx in range(ns):
#       for t_idx,tstep in enumerate([0,2,3]):
#         arscores[curr_idx,seed_idx,t_idx] = adjusted_rand_score(
#           exp_batch_data[curr_idx][seed_idx]['exp'][:,1],
#           exp_batch_data[curr_idx][seed_idx]['zt'][:,tstep]
#         )
#   return arscores

# def count_num_schemas(exp_data):
#   """ 
#   """
#   L = []
#   for curr_idx in range(len(condL)):
#     num_schemas_used = [
#       len(np.unique(exp_data[curr_idx][i]['zt'][:,:-1].flatten())
#          ) for i in range(ns)
#     ]
#     L.append(num_schemas_used)
#   nschemas = np.array(L)
#   return nschemas

## acc
# batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
## extract others
# arscores = calc_adjrand(exp_batch_data) # 
# nschemas = count_num_schemas(exp_batch_data) # curr,nseeds
# zt = np.array(
#   [[exp_batch_data[j][i]['zt'] for i in range(ns)] for j in range(5)]
# ) # curr,seed,trial,tstep
