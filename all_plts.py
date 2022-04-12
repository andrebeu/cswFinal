
import sys
import os, re
import pandas as pd
import numpy as np
import json

from matplotlib import pyplot as plt


def get_sql_df(exp_version=None):
  import sqlite3 as sql
  from glob import glob as glob
  db_fpath = 'csw_mturk_spring19.db'
  with sql.connect(db_fpath) as conn:
    # connection objecs represent the database
    # cursor objects point to rows in the database
    c = conn.cursor()

    # list tables in database
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables_in_db = c.fetchall()
    table_name = "CSWfall18" # table name must be constant to prevent repeat subjects

    # table header
    c.execute("PRAGMA table_info(%s)"%table_name)
    col_names = [i[1] for i in c.fetchall()]

    # select everything within table
    db_datastring = c.execute("""SELECT * FROM %s"""%table_name).fetchall()
    sql_df = pd.DataFrame(db_datastring,columns=col_names)
    
    # remove unneeded rows 
    sql_df = sql_df[sql_df['mode'] != 'debug'] # debug rows
    sql_df = sql_df[pd.notnull(sql_df['datastring'])] # rows with no datastring 
    # return only select code version
    versions = sql_df.codeversion.unique()
    print('exp versions found in table:')
    for v in versions: print(v)
    # if version not specified, take latest
    if exp_version==None:
      exp_version = versions[-1]
    print('\n-loading code version:',exp_version)
    sql_df = sql_df[sql_df.codeversion == exp_version]
    print('N =',len(sql_df),'subjects')
  return sql_df
_ = get_sql_df()


# '40 blocked': '1000clq',
# 'interleaved': 'i1000cl',

# In[3]:


CONDITION = str(sys.argv[1])

if CONDITION == 'interleaved':
  dataset_code = 'i1000cl'
elif CONDITION == 'interleaved_rep':
  dataset_code = 'csw1000block01.04.25.19'
elif CONDITION == 'blocked':
  dataset_code = '1000clq'
elif CONDITION == 'blocked_rep':
  dataset_code = 'csw1000block40.04.07.19'
elif CONDITION == 'explicit_interleaved':
  dataset_code = 'csw1000boldmdp.05.09.19'
elif CONDITION == 'inserted_early':
  dataset_code = 'csw1000insertedblock0.09.22.19'
elif CONDITION == 'inserted_early_rep':
  dataset_code = 'csw1000insertedblock0.10.08.19'
elif CONDITION == 'inserted_middle':
  dataset_code = 'csw1000insertedblock1.09.30.19'
elif CONDITION == 'inserted_middle_rep':
  dataset_code = 'csw1000insertedblock1.10.07.19'
elif CONDITION == 'inserted_late':
  dataset_code = 'csw1000insertedblock2.10.09.19'
elif CONDITION == 'inserted_late_rep':
  dataset_code = 'csw1000insertedblock2.10.10.19'
else:
  print("INVALID CONDITION",CONDITION)
  assert False


# In[5]:


sql_df = get_sql_df(dataset_code)
sql_df = sql_df # keep first 50 only
Nsubs,_ = sql_df.shape


# In[6]:


sql_df


# ### comments

# In[7]:


# I = -5 # -3,-5
# for i in range(len(sql_df)):
#   print(i)
#   resp = json.loads(
#           pd.DataFrame(
#             json.loads(
#               sql_df.iloc[i]['datastring'])['data']
#           ).iloc[I].loc['trialdata']['responses'])['Q0']
#   if len(resp):
#     print('--\n')
#     print(resp)


# ### make dataframe

# In[8]:


def get_block_indices(subj_df,num_stories=110,num_blocks=11):
  """ given a subj_df
  returns the indices of when blocks begins and ends
  which can be used to index a block of stories in subj_df 
  """
  # get indices for begining of stories
  begin_story_bool = subj_df.state == 'BEGIN'
  begin_story_idx = subj_df[begin_story_bool].index
  # get indices for begin and end of blocks
  block_linsp = np.arange(0,109,10)
  begin_block_idx = begin_story_idx[block_linsp]
  end_block_idx = np.hstack([begin_block_idx[1:],np.array([len(subj_df)])])
  return begin_block_idx,end_block_idx

def include_block_idx(subj_df):
  """ inplace column
  """
  subj_df['block'] = 0
  begin_block_idx,end_block_idx = get_block_indices(subj_df)
  for block_num,(bidx,eidx) in enumerate(zip(begin_block_idx,end_block_idx)):
    subj_df.loc[bidx:eidx,'block'] = int(block_num)
  return subj_df

def include_story_idx(subj_df,num_stories):
  """ inplace include column with stories
  """
  bidx = subj_df.loc[subj_df['state']=='BEGIN'].index
  subj_df.loc[bidx,'story'] = np.arange(num_stories,dtype=np.int)
  subj_df.story = np.floor(subj_df.story.interpolate()).astype('int')
  return subj_df


# In[9]:


frnode2depth = lambda x: DEPTH_DICT[x]
DEPTH_DICT = {np.nan:100,'BEGIN':0,
              'NODE11':1,'NODE12':1,
              'NODE21':2,'NODE22':2,
              'NODE31':3,'NODE32':3,
              'END':4}

def make_subj_df(psiturk_dstr,num_instruction_trials=3,stories_per_block=40):
  # initialize
  subj_df = pd.DataFrame([t['trialdata'] for t in json.loads(psiturk_dstr)['data']])
  # remove instruction trials
  subj_df = subj_df.iloc[num_instruction_trials:]
  # subj response
  subj_df['response_right'] = subj_df['key_press'] == 39.0 
  subj_df['true_on_right'] = (subj_df['true_on_right'] == "True")
  subj_df['correct_response'] = (subj_df['true_on_right'] == subj_df['response_right'])
  # only keep some columns
  subj_df = subj_df[['type','qtype',
                     'state','fromnode','true_tonode','false_tonode',
                     'rfc','true_rfc','false_rfc',
                     'response_right','true_on_right','correct_response',]]
  ## customizing
  # story columns
  subj_df = include_story_idx(subj_df,num_stories=200)
  # edges
  subj_df['edge'] = subj_df['fromnode'] + subj_df['true_tonode']
  # blocks
  subj_df['block'] = subj_df.story//stories_per_block
  # depth
  subj_df['depth']=subj_df.fromnode.apply(frnode2depth)
  return subj_df

psiturk_dstr = sql_df['datastring'].iloc[0] 
subj_df = make_subj_df(psiturk_dstr)
subj_df


# ### make subject and group dataframes 

# In[10]:


def make_group_df(sql_df):
  """ looping `make_subj_df`
  """
  subj_df_L = []
  # loop making subject dfs
  for s in range(len(sql_df)):
    psiturk_dstr = sql_df['datastring'].iloc[s] 
    subj_df = make_subj_df(psiturk_dstr)
    subj_df['subjnum']=s
    subj_df_L.append(subj_df)
  # make group dfs
  group_df = pd.concat(subj_df_L)
  # reshape
  group_df = group_df.set_index(['subjnum','block','story','depth',group_df.index])
  return group_df

group_df = make_group_df(sql_df)
group_df


# ### split dataframes GLOBAL VARIABLES

# In[11]:


# should i make these global vars?
# i.e. access them within funs without passing
_,_,(_,group_qdf),(_,group_sdf) = group_df.groupby('type')
(_,group_fqdf),(_,group_tqdf) = group_qdf.groupby('qtype')
group_tqdf


# In[12]:


def get_thresholded_tqdf(group_fqdf=group_fqdf,threshold=0.8):
  """ takes in dataframe
  returns dataframe
  """
  # thresholding
  sub_mask = group_fqdf.correct_response.groupby('subjnum').mean() > threshold
  sub_mask = sub_mask[sub_mask].index.values
  masked_group_df = group_tqdf.loc[sub_mask,:,:]
  return masked_group_df

def select_transition_probes(group_tqdf):
  """ remove transition from BEGIN node"""
  return group_tqdf[group_tqdf['fromnode']!='BEGIN']
  


# # plotting

# In[13]:


## threshold based on filler questions
THRESHOLD = 0.9
plt_group_tqdf = get_thresholded_tqdf(threshold=THRESHOLD)
## remove begin node probe (only happens during training)
plt_group_tqdf = select_transition_probes(plt_group_tqdf)


# In[14]:


num_subs = len(plt_group_tqdf.index.get_level_values('subjnum').unique())

# plotting
grouping = 'story'
plt.figure(figsize=(15,4))
M = plt_group_tqdf.correct_response.groupby([grouping]).mean()
N = plt_group_tqdf.correct_response.groupby([grouping]).count()
S = plt_group_tqdf.correct_response.groupby([grouping]).std()/np.sqrt(N)
# plt.bar(np.arange(len(M))+1,M,yerr=S)
plt.plot(np.arange(len(M))+1,M,c='purple',lw=3)
plt.fill_between(np.arange(1,201),M-S,M+S,
                  facecolor='purple',alpha=0.5)

plt.axhline(0.5,c='lime',ls='--')
plt.yticks(np.arange(0,1.1,.1))

plt.ylim(0.2,1.01)
plt.grid(axis='y')

block_size = 40
if CONDITION == 'blocked' or CONDITION == 'blocked_rep':
  cL = ['green','blue','green','blue']
elif CONDITION == 'interleaved' or CONDITION == 'interleaved_rep' or CONDITION == 'explicit_interleaved':
  cL = ['yellow']
  block_size = 160
elif CONDITION == 'inserted_early' or CONDITION == 'inserted_early_rep':
  cL = ['blue','green','purple','purple']
elif CONDITION == 'inserted_middle' or CONDITION == 'inserted_middle_rep':
  cL = ['purple','blue','green','purple']
elif CONDITION == 'inserted_late' or CONDITION == 'inserted_late_rep':
  cL = ['purple','purple','blue','green']

for idx,t1 in enumerate(range(0,160,block_size)):
  t2 = t1+block_size+1
  plt.fill_between(np.arange(t1,t2),0,1.05,
                  facecolor=cL[idx],alpha=0.1)
plt.fill_between(np.arange(160,201),0,1.05,
                  facecolor='red',alpha=0.2)

#   plt.axvline(t1,c='r',alpha=.5)
  
plt.xlim(0,200)
print(num_subs,Nsubs)
title = 'group_%s-avgwi_%s-N_%i(%i)'%(CONDITION,grouping,num_subs,Nsubs)
plt.title(title)
plt.savefig('figures/csw_plts_2020/group/'+title)
# plt.savefig('../presentations/psychonomics19/'+title,format='svg')


# # subjects plot

# In[15]:


def mov_avg(A,wind):
  N = len(A)-wind
  M = -np.ones([N])
  for t in range(N):
    M[t] = A[t:t+wind].mean()
  return M


# In[31]:


for MA_WINDOW in [1,5,20]:
  subj_data = plt_group_tqdf.correct_response.groupby(['subjnum'])
  subj_data_plt = subj_data.apply(mov_avg,MA_WINDOW)

  plt.close('all')
  fig,axarr = plt.subplots(10,5,figsize=(30,30));axarr=axarr.reshape(-1)
  for plt_idx,sub_idx in enumerate(subj_data_plt.index):
    ax = axarr[plt_idx]
    sub_ma = subj_data_plt[sub_idx]
    ax.scatter(range(len(sub_ma)),sub_ma,c='purple',lw=3)
    ax.axvline(int(len(sub_ma)*(4/5))-15)
    ax.set_xticks([])
    ax.set_ylim(-0.05,1.05)
    ax.axhline(0.5,c='r',ls='--')
    ax.axhline(1,c='b',ls='--')

  title = 'subs_%s_window_%i'%(CONDITION,MA_WINDOW)
  plt.savefig('figures/csw_plts_2020/subject/'+title)


# In[ ]:




