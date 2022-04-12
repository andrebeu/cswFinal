# this file imports custom routes into the experiment server

from flask import Blueprint, render_template, request, jsonify, Response, abort, current_app

from psiturk.psiturk_config import PsiturkConfig
from psiturk.experiment_errors import ExperimentError
from psiturk.user_utils import PsiTurkAuthorization

# # Database setup
from psiturk.db import db_session
from psiturk.models import Participant
from json import loads

# load the configuration options
config = PsiturkConfig()
config.load_config()
myauth = PsiTurkAuthorization(config)  # if you want to add a password protect route use this
custom_code = Blueprint('custom_code', __name__, template_folder='templates', static_folder='static')


### CUSTOM

import pandas as pd
import numpy as np

MIN_BONUS = 2.00
MAX_EXTRA_BONUS = 2.00

#----------------------------------------------
# Computing bonus
#----------------------------------------------


def calc_subj_filler_score(subjectObj):
  subj_data_json = [i['trialdata'] for i in loads(subjectObj.datastring)['data']]
  subj_df = pd.DataFrame(subj_data_json)
  subj_df['response_right'] = subj_df['key_press'] == 39.0 
  subj_df['true_on_right'] = (subj_df['true_on_right'] == "True")
  subj_df['correct_response'] = (subj_df['true_on_right'] == subj_df['response_right'])
  subj_fillerdf = subj_df[subj_df['qtype'] == 'fillerQ']
  percent_correct = float(sum(subj_fillerdf['correct_response']))/len(subj_fillerdf)
  return percent_correct



@custom_code.route('/compute_bonus', methods=['GET'])
def compute_bonus():
  # check that user provided the correct keys
  # errors will not be that gracefull here if being
  # accessed by the Javascrip client
  if not request.args.has_key('uniqueId'):
    raise ExperimentError('improper_inputs')  
  uniqueId = request.args['uniqueId']

  try:
    # lookup user in database
    user = Participant.query.filter(Participant.uniqueid == uniqueId).one()

    subj_filler_score = calc_subj_filler_score(user)

    ## bonus calculations
    subj_bonus = MIN_BONUS
    # performance extra
    if subj_filler_score >= 0.9:
      subj_bonus += MAX_EXTRA_BONUS*subj_filler_score

    user.bonus = np.round(subj_bonus,2)
    db_session.add(user)
    db_session.commit()
    resp = {"bonusComputed": "success"}
    return jsonify(**resp)
  except:
    abort(404)  # display HTML
