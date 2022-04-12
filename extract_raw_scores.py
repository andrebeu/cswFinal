#!/usr/bin/env python
# coding: utf-8

import os, re
import pandas as pd
import numpy as np
import json

from csw2022 import *


for condition in conditionL:
    ## loading
    # load psiturk `experiment dataframe`
    sql_df = get_sql_df(get_dataset_code(condition))
    # from `experiment dataframe`, pull data pertaining to subjects responses
    subj_df = make_subj_df(sql_df['datastring'].iloc[0])
    # reindex dataframe to make easy to read
    group_df = make_group_df(sql_df)

    ## processing
    # breakup dataframe by story and question nodes
    _,_,(_,group_qdf),(_,group_sdf) = group_df.groupby('type')
    # breakup question dataframe by filler and transition nodes
    (_,group_fqdf),(_,group_tqdf) = group_qdf.groupby('qtype')


    ## threshold based on filler questions
    plt_group_tqdf = get_thresholded_tqdf(group_tqdf,group_fqdf,THRESHOLD)
    # remove begin node probe (only happens during training)
    plt_group_tqdf = select_transition_probes(plt_group_tqdf)

    ## make long dataframe and export

    long_df = plt_group_tqdf.groupby(['depth','subjnum','story']
              ).mean().correct_response.reset_index().astype(int)
    long_df.to_csv('experiment_data/raw_scores/%s.csv'%condition)
