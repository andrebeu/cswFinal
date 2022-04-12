import os, re
import pandas as pd
import numpy as np
import json

## load data

def get_sql_df(exp_version=None,verb=True):
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
    if verb:
      print('exp versions found in table:')
    for v in versions: 
      if verb: print(v)
    # if version not specified, take latest
    if exp_version==None:
      exp_version = versions[-1]
    
    sql_df = sql_df[sql_df.codeversion == exp_version]
    if verb:
      print('\n-loading code version:',exp_version)
      print('N =',len(sql_df),'subjects')
  return sql_df

## make df

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

def get_thresholded_tqdf(group_tqdf,group_fqdf,threshold=0.9):
  """ takes in dataframe
  returns dataframe
  """
  # thresholding
  sub_mask = group_fqdf.correct_response.groupby('subjnum').mean() > threshold
  sub_mask = sub_mask[sub_mask].index.values
  masked_group_df = group_tqdf.loc[sub_mask,:,:]
  return masked_group_df


## analysis

def extract_subj_test_score(sql_df,thresh=0.9):
  group_df = make_group_df(sql_df)
  _,_,(_,group_qdf),(_,group_sdf) = group_df.groupby('type')
  (_,group_fqdf),(_,group_tqdf) = group_qdf.groupby('qtype')
  # tqdf > thresh
  masked_group_tqdf = get_thresholded_tqdf(group_tqdf,group_fqdf,threshold=thresh)
  num_subs = len(masked_group_tqdf.index.get_level_values('subjnum').unique())
  # select test scores
  data = -np.ones(num_subs)
  for idx,(_,subj_qdf) in enumerate(masked_group_tqdf.correct_response.groupby('subjnum')):
    data[idx] = np.mean([i[1].mean() for i in subj_qdf.groupby('story')][-40:])
  return data
  






