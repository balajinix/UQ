"""
  File:             evaluation.py

  About
  ------------------
  evaluation of UQ work

"""

import numpy as np
import pandas as pd
import os
import json
import csv
import random

# import scripts that are needed
import gen_calibration_error
#import visualizer

from sklearn import metrics

from nltk.tokenize import wordpunct_tokenize # used for tokenization for SQL type

from rouge_score import rouge_scorer # used for computing rouge score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)




# function computes rejection-VALUE curve, where VALUE could be accuracy, etc.
# here acc is a list of accuracies (but it cud be any metric, since AUARC works for any metric)
# conf is a corresponding list of confidences -- again this cud in theory be any score, not necessarily b/w 0 and 1

def compute_auarc(acc, conf):

  # create pandas dataframe and do a random shuffle of rows due to the issue with AUARC around identical confidences
  df = pd.DataFrame({"c": conf, 'a': acc}).sample(frac=1)
  # df = pd.DataFrame({"c": conf, 'a': acc}) -- this is the version w/o random shuffle
  # sort by confidences and compute the metric
  df = df.sort_values('c', ascending=False)  # check if ascending should be true or false!!!!
  df['amean'] = df['a'].expanding().mean()
  auarc = metrics.auc(np.linspace(0, 1, len(df)), df['amean'])

  return auarc


