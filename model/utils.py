import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from itertools import product
import time
import seaborn as sns
sns.set_context('talk')

from model import *
## import human data for fitting
hdf = pd.read_csv('../human_data.csv')
humanB_acc,humanI_acc = hdf.loc[:,('blocked mean','interleaved mean')].values.T

FLAG_SMACC = True
SMTEMP = 4  

def param2str(args):
    param_str = "-".join(["%s_%.3f"%(i,j) for i,j in args['sch'].items()])
    param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in args['sem'].items()])
    return param_str

def softmax_custom(x,tau):
    return np.exp(x*tau)/np.sum(np.exp(x*tau))

def get_sm(xth,norm=True):
  """ 
  given x_t_hat from subject
  [trial,layer,node]
  get 2afc normalized softmax for layer 2/3
  return: [layer2/3,trial,node56/78]
  norm=true 
   apply softmax to xth
   when prediction done with multiple schemas
  """
  nodes = {2:(5,6),3:(7,8)} 
  L = [] # layer 2 and 3
  for l,ns in nodes.items():
    y = xth[:,l,ns]
    # print(y.shape)
    # assert False
    if norm:
      # y = softmax(y,1)
      y = np.array([softmax_custom(yt,SMTEMP) for yt in y])
    L.append(y)
  return np.array(L)

def get_acc(data,acc_mode=FLAG_SMACC):
    """ 
    returns 2afc softmax of 
    layer 2/3 transitions
    single seed
    """
    if acc_mode: # compute softmax acc
        yhat_sm = get_sm(data['xth'])
        # print(data['xth'],yhat_sm)
        L = []
        # loop over layers
        for i in range(2):
            yhat_sm_l = yhat_sm[i,:,:] # softmax of layer (float)
            yt = data['exp'][:,i+3] # target tonode (int)
            pr_yt = yhat_sm_l[range(len(yhat_sm_l)),yt - (5+2*i)] # 
            L.append(pr_yt)
        return np.array(L)
    else: # compute score
        xth = data['xth']
        resp = xth.argmax(-1)
        exp = data['exp']
        score = resp[:,2:4] == exp[:,3:5]
        # transpose required for backcompatibility
        return score.T


def unpack_acc(cbatch_data,mean_over_tsteps=True):
    """ 
    given cbatch data (data from multiple curr and seeds)
    return acc [curr,seed,trial]
    """
    accL = [] # curr
    for cidx in range(len(cbatch_data)):
        acc = np.array([get_acc(sbatch) for sbatch in cbatch_data[cidx]])
        if mean_over_tsteps:
            accL.append(acc.mean(1)) # mean over layers
        else:
            accL.append(acc) # mean over layers
    return np.array(accL)

def unpack_data(cbatch_data,dtype='priors'):
    """ unpacks batch data from multiple curr and seeds
    dtype: priors,likes,post
    """
    L = []
    for cidx in range(len(cbatch_data)):
        L.append([])
        for sidx,sbatch_data in enumerate(cbatch_data[cidx]):
            # print(sidx,sbatch_data[dtype].shape)
            # assert False
            mask = np.any(sbatch_data[dtype]!=-1,0)[0]
            # print(mask)
            L[cidx].append(sbatch_data[dtype][:,:,mask])
    return L


### RUN EXP


def run_batch_exp_curr(ns,args,currL=['blocked','interleaved']):
  """ loop over task conditions, 
  returns list of list of dicts
    data[curr][seed] = 
        dict_keys(['zt', 'xth', 'prior', 'like', 'post', 'exp'])
  """
  accL = []
  dataL = []
  # dataD = {}
  for currName in currL:
    args['exp']['condition'] = currName
    ## extract other data here
    data_batch = run_batch_exp(ns,args)
    dataL.append(data_batch)
    # dataD[curr] = dataL
    ## unpack seeds and take mean over layers
    acc = np.array([get_acc(data) for data in data_batch]).mean(1) # mean over layer
    accL.append(acc)
  return dataL
  
def run_batch_exp(ns,args,expArr=None):
  """ exp over seeds, 
  single task_condition / param config
  return full data
  """
  dataL = []
  for i in range(ns):
    task = Task()
    sem = SEM(schargs=args['sch'],**args['sem'])
    if type(expArr) == type(None):
        exp,_  = task.generate_experiment(**args['exp'])
    else:
        exp = expArr[i]
    data = sem.run_exp(exp)
    data['exp']=exp
    dataL.append(data)
  return dataL




## plotting

def movavg(X,w=5):
  Y = np.zeros(len(X)-w)
  for xi in range(len(X)-w):
    Y[xi] = X[xi:xi+w].mean()
  return Y
    

## TOP MSE models
schargs_mse0 ={
  'concentration': 1.524765,
  'stickiness_wi': 1.083961,
  'stickiness_bt': 1.083961,
  'sparsity': 0.152714
}
schargs_mse1 ={
  'concentration': 1.500351,
  'stickiness_wi': 1.375400,
  'stickiness_bt': 1.375400,
  'sparsity': 0.112368
}
schargs_mse2 ={
  'concentration': 0.492800,
  'stickiness_wi': 1.287879,
  'stickiness_bt': 1.287879,
  'sparsity': 0.039337
}
schargs_mse3 ={
  'concentration': 0.745254,
  'stickiness_wi': 1.226039,
  'stickiness_bt': 1.226039,
  'sparsity': 0.005049
}
schargs_mse4 ={
  'concentration': 2.070620,
  'stickiness_wi': 1.277997,
  'stickiness_bt': 1.277997,
  'sparsity': 0.004873
}
SCHARGS_MSE = [schargs_mse0,schargs_mse1,schargs_mse2,schargs_mse3,schargs_mse4]