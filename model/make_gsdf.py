import numpy as np
from glob import glob as glob
import pandas as pd

## import human data for fitting
hdf = pd.read_csv('../human_data.csv')
hB,hI = hdf.loc[:,('blocked mean','interleaved mean')].values.T
hdf_split = pd.read_csv('../human_data_split_tstep.csv')
hB1,hB2,hI1,hI2 = hdf_split.loc[:,('blocked-step1','blocked-step2',
    'interleaved-step1','interleaved-step2')].values.T
    
## set
gsname = 'gs0317'
MAKE_DATADF = False
MAKE_SUMMDF = True
INNER_SIZE = 500
J_ = 2

if MAKE_DATADF:
    for jdx in range(1*J_):
        print('jdx',jdx)
        dfpathL = glob('data/%s/*'%gsname)
        datadfL = []
        for idx in np.arange(jdx*1000+INNER_SIZE,jdx*1000+1001,INNER_SIZE):
            print(idx-INNER_SIZE,idx)
            ## inner
            dfL = []
            for p in dfpathL[idx-INNER_SIZE:idx]:
                try:
                    df_ = pd.read_csv(p)
                    dfL.append(df_)
                except:
                    print('error',p)
                    continue
            print('inner concating. len=',len(dfL))
            if len(dfL):
                mini_datadf = pd.concat(dfL).drop(columns='Unnamed: 0',)
                datadfL.append(mini_datadf)
        print('outer concating')
        if len(datadfL):
            datadf = pd.concat(datadfL)
            # datadf ## full data
            print('saving datadf')
            datadf.to_csv('data/%s-datadf-%i.csv'%(gsname,jdx))
            del datadf
else:
    ### 
    None
    # print('reading datadf')
    # datadf = pd.read_csv('data/%s-datadf.csv'%gsname)

# datadf.columns
paramL = ['concentration', 'stickiness_wi', 'stickiness_bt',
   'sparsity', 'pvar', 'lrate', 'lratep', 'decay_rate',
   'skipt1']

## exp df
def make_exp_summ_df(exp_data_df,split_acc_bool=True):
    """ 
    given data_df for a collection of params
        groups by params and cond to compute 
        relevant metrics e.g. test acc
    returns a one row dataframe with 
        the results of an exp of params
        i.e. params in single file
    """
    seed_summ_df_L = []
    dfgrp = exp_data_df.groupby(paramL)
    for params_i,seed_df in dfgrp:
        dataD = {**dict(zip(paramL,params_i))}
        # loop conditions (BIEML)
        for cond_i,df_c in seed_df.groupby('cond'):
            # print
            if len(df_c.trial)!=200: 
                print('skip')
                continue
            # MSE
            if split_acc_bool:
                df_c.loc[:,"acc"] = (df_c.acc2+df_c.acc3)/2
                if (cond_i == "blocked") or (cond_i == "interleaved"):
                    hacc_step1 = hdf_split.loc[:,("%s-step1"%cond_i)].values.T
                    hacc_step2 = hdf_split.loc[:,("%s-step2"%cond_i)].values.T
                    mse1 = np.mean((hacc_step1-df_c.acc2)**2)
                    mse2 = np.mean((hacc_step2-df_c.acc3)**2)
                    dataD['%s_mse1'%cond_i[0]] = mse1
                    dataD['%s_mse2'%cond_i[0]] = mse2
                # dataD['acc-%s'%cond_i[0]] = df_c.acc
            else:
                ## compute metrics
                testacc = np.mean(df_c.acc[-40:])
                acc2 = np.mean(df_c.acc[40:80])
                hdata = hdf.loc[:,('%s mean'%cond_i)].values.T
                MSE = np.mean((hdata-df_c.acc)**2)            
                ## populate dataD
                dataD['mse-%s'%cond_i[0]] = MSE
                dataD['acc2-%s'%cond_i[0]] = acc2
                dataD['testacc-%s'%cond_i[0]] = testacc
        # assert False
        ##
        seed_summ_df_L.append(pd.DataFrame(index=[0],data=dataD))
    return pd.concat(seed_summ_df_L)


## NEW STRATEGY: does not require intermediate datadf
if MAKE_SUMMDF:
    print('make summary df')
    dfpathL = glob('data/%s/*'%gsname)
    mini_summdf_L = []
    for idx in np.arange(INNER_SIZE,1000*J_+1,INNER_SIZE):
        print(idx-INNER_SIZE,idx)
        ## inner
        exp_summ_df_L = []
        for p in dfpathL[idx-INNER_SIZE:idx]:
            try:   
                # read an expdf 
                # (i.e. full trace of tens of params)
                exp_data_df = pd.read_csv(p)
            except:
                print('error reading',p)
                continue
            ## make summary df from expdf 
            exp_summ_df = make_exp_summ_df(exp_data_df)
            exp_summ_df_L.append(exp_summ_df)
        ## 
        if len(exp_summ_df_L):
            print('inner concat. summ df from a collection of exps')
            mini_summ_df = pd.concat(exp_summ_df_L)
            print('should contain 4 large summ_dfs')
            mini_summdf_L.append(mini_summ_df)
    print('outer concat')
    gsdf = pd.concat(mini_summdf_L)
    gsdf.to_csv('data/%s-summdf.csv'%gsname)





### old strategy fails because large data volumes
if False:
    print('making summdf')
    ## compute summary df
    groupvars = paramL 
    gsdf_group = datadf.groupby(groupvars)
    del datadf
    print('grouped')
    dfL = []
    ### BOTTLENECK GSDF_GROUP IS TOO LARGE TO UNPACK
    for params_i,df_i in gsdf_group:
      print(params_i)
      dataD = {**dict(zip(groupvars,params_i))}
      ## loop conditions (BIEML)
      for cond_i,df_c in df_i.groupby('cond'):
        ## compute metrics
        testacc = np.mean(df_c.acc[-40:])
        ## populate dataD
        dataD['testacc-%s'%cond_i[0]] = testacc
      
      print('append') 
      dfL.append(pd.DataFrame(index=[0],data=dataD))
      ##
    print('cating')
    gsdf = pd.concat(dfL)
    print('saving summdf')
    gsdf.to_csv('data/%s-summdf.csv'%gsname)



print('done')