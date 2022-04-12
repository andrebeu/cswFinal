import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from itertools import product
import time
import seaborn as sns
sns.set_context('talk')
from sklearn.metrics import adjusted_rand_score

from model import *
from utils import unpack_acc
## import human data for fitting
hdf = pd.read_csv('../human_data.csv')
humanB_acc,humanI_acc = hdf.loc[:,('blocked mean','interleaved mean')].values.T


condL = ['blocked','interleaved',
         'early','middle','late'
        ]


## get params
def get_argsD(row):
  """ 
  takes dataframe row,
  returns sem args 
  """
  ## params
  taskargs = {
    'condition':None, # defined in loop
    'n_train':160,
    'n_test':40
  }
  semargs = {
    'beta2':0,
    'skipt1':1,
    'ppd_allsch':0
  }
  ## get params from top MSE
  schargs = {
     'concentration':dict(row)['concentration'],
     'stickiness_wi':dict(row)['stickiness_wi'],
     'stickiness_bt':dict(row)['stickiness_bt'],
     'sparsity':dict(row)['sparsity'],
     'pvar': 0,
     'lrate':1,
     'lratep':1,
     'decay_rate':1,
  }
  args = {
      'sem':semargs,
      'sch':schargs,
      'exp':taskargs
  }
  param_str = "-".join(["%s_%.3f"%(i,j) for i,j in schargs.items()])
  param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in semargs.items()])
  return args,param_str

# PLOT ACC
def plt_acc(exp_batch_data):
  batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
  mean_acc = batch_acc.mean(1)
  plt.figure(figsize=(20,10))
  for idx in range(len(condL)):
    plt.plot(mean_acc[idx],label=condL[idx])
  plt.legend()
  plt.axhline(0.5,c='k',ls='--')
  plt.ylim(-0.05,1.01)
  # plt.title(param_str)
  plt.grid(True,axis='y')
  
## VIOLIN 
## count number of schemas used
def count_num_schemas(exp_batch_data):
  """ 
  """
  nseeds = len(exp_batch_data[0])
  L = []
  for curr_idx in range(len(condL)):
    num_schemas_used = [
      len(np.unique(exp_batch_data[curr_idx][i]['zt'][:,:-1].flatten())
         ) for i in range(nseeds)
    ]
    L.append(num_schemas_used)
  nschemas = np.array(L)
  return nschemas


def plt_LC_violins(exp_batch_data):
  nschemas = count_num_schemas(exp_batch_data)
  M = nschemas.mean(1)
  S = nschemas.std(1)
  plt.figure(figsize=(20,10))
  plt.title('number schemas used')
  plt.violinplot(nschemas.T,np.arange(len(condL)),showmeans=True)
  ax = plt.gca()
  ax.set_xticks(range(len(condL)))
  ax.set_xticklabels(condL)
  plt.grid(True,axis='y')
  # plt.title(param_str)

## ADJ RAND
def calc_adjrand(exp_batch_data):
  nseeds = len(exp_batch_data[0])
  arscores = -np.ones([len(condL),nseeds,3])
  for curr_idx in range(len(condL)):
    for seed_idx in range(nseeds):
      for t_idx,tstep in enumerate([0,2,3]):
        arscores[curr_idx,seed_idx,t_idx] = adjusted_rand_score(
          exp_batch_data[curr_idx][seed_idx]['exp'][:,1],
          exp_batch_data[curr_idx][seed_idx]['zt'][:,tstep]
        )
  return arscores

def plt_arscores(exp_batch_data):
  nseeds = len(exp_batch_data[0])
  arscores = calc_adjrand(exp_batch_data)
  f,axar=plt.subplots(1,3,figsize=(30,6),sharey=True)
  for t in range(3):
    ax=axar[t]
    ax.violinplot(arscores[:,:,t].T,showextrema=1,showmeans=1)
    for c in range(len(condL)):
      ax.scatter(np.repeat(c+1,nseeds),arscores[c,:,t].T)
    ax.set_xticks(np.arange(1,len(condL)+1))
    ax.set_xticklabels(condL)
    ax.set_title(['0','2','3'][t])
  # plt.suptitle(param_str)