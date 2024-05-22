"""
  File:             uq_from_similarity.py

  About
  ------------------
  methods for assessing similarity and estimating confidence from similarities

"""

import numpy as np
import os
import json
#import pandas as pd
#import csv
#import random

# import scripts that are needed
import gen_calibration_error
#import visualizer

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import svm

from nltk.tokenize import wordpunct_tokenize # used for tokenization for SQL type

from rouge_score import rouge_scorer # used for computing rouge score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from sql_metadata import Parser # used for computing Aligon score

from sentence_transformers import SentenceTransformer # used for computing sBERT
SBERT_MODEL_NAME = "all-MiniLM-L12-v2" # this is supposedly 5 times faster than "all-mpnet-base-v2"
#SBERT_MODEL_NAME = "all-mpnet-base-v2" # this takes a while
#SBERT_MODEL_NAME = "bert-base-nli-mean-tokens" # this does not seem to be faster than "all-MiniLM-L12-v2"
#SBERT_MODEL_NAME = "average_word_embeddings_glove.840B.300d" # non-transformer embedding; this seems slow

#PRECOMPUTED_SIMS_CASE = 'sim_dict_spider_dev_codellama_temp_first'
#PRECOMPUTED_SIMS_CASE = 'sim_dict_spider_dev_deepseeker_temp_first'
#PRECOMPUTED_SIMS_CASE = 'sim_dict_spider_dev_granite_temp_first'
PRECOMPUTED_SIMS_CASE = 'sim_dict_spider_realistic_dev_codellama_temp_first'

PRECOMPUTED_SIMS = ['sbert', 'aligon', 'aouiche', 'makiyama']
#PRECOMPUTED_SIMS = ['aligon', 'aouiche', 'makiyama']

from scipy.stats import beta
from scipy.special import logit


# dictionary where key is query index and value is list of confidences for each sample
# note that this needs to be later re-configured by sample index rather than query index for evaluation
def prepare_conf_dict(samples_data, sim_type, sim_dict, uq_type, bayes_param_dict, clf, eps):

  # PREPARE SIM DICT FROM POTENTIALLY PRE-COMPUTED SIMILARITIES (if not already computed)
  if sim_type in PRECOMPUTED_SIMS and not sim_dict:

    sim_dict = prepare_sim_dict(sim_type)
    print('Finished preparing similarity dict!')

  conf_dict = {}

  query_index = 0
  for query in samples_data:
    samples = query['samples']
    gen_sql_list = [sample['gen_sql'] for sample in samples]

    # formulate similarity matrix -- if possible, prepare it from precomputed similarities
    if sim_type in PRECOMPUTED_SIMS:
      question = query['question']
      W = get_W_from_precomputed_sims(gen_sql_list, question, sim_dict)
    else:
      W = get_W(gen_sql_list, sim_type) # -- SBERT model is hardcoded in this function
    #print(W)

    # make matrix non-singular for 'spec' approach -- hardcoding increment in the diagonal entries
    # not needed anymore since we fixed the diagonal bug
    # if 'spec' in uq_type:
    #   eps_increment = 0.000001
    #   for i in range(len(W[0,:])):
    #     W[i,i] += eps_increment

    if 'agg' in uq_type:
      agg_type = uq_type.split('-')[1]
      conf_list = sim_to_conf_aggregation(W, agg_type)
    elif 'spec' in uq_type:
      metric = uq_type.split('-')[1]
      thres = 0.9
      conf_list = get_conf_spectral_clustering(W, metric, thres, None)
    elif 'bayes' in uq_type:
      bayes_type = uq_type.split('-')[1]
      conf_list = conf_estimation_bayes_single_query(W, bayes_type, bayes_param_dict, eps)
    elif 'clf' in uq_type:
      conf_list = conf_estimation_clf_single_query(W, clf, eps)
    else:
      raise ValueError(f'Unknown UQ type specified {uq_type}')

    conf_dict[query_index] = conf_list
    # evaluating rouge and sbert takes time, so keep track of confidence estimation for queries by printing
    batch_size = 100
    if 'rouge' in sim_type or sim_type == 'sbert':
      if (query_index+1) % batch_size == 0:
        print('Computed confidences up to query:', query_index + 1)

    query_index += 1

  return conf_dict


################################
# SIMILARITY AGGREGATION
################################

# similarity aggregation approach for going from similarities to confidences
def sim_to_conf_aggregation(W, agg_type):

  num_samples = len(W[0,:])

  conf_list_over_samples = [0] * num_samples # initialize to all 0s
  for i in range(num_samples):

    dist_vec = 1 - W[i,:]

    if agg_type == 'arith': # arithmetic mean
      agg_dist = np.mean(dist_vec)
    elif agg_type == 'geom': # geometric mean
      #print('len(dist_vec):', len(dist_vec))  # -- for some reason, not working with sbert for codellama?
      #print(dist_vec)
      agg_dist = np.array(dist_vec).prod() ** (1.0 / len(dist_vec))
    elif agg_type == 'harm': # harmonic mean
      eps_harm = 0.000001 # hardcoded jitter for computing harmonic mean by preventing 0 distance
      dist_vec = [dist + eps_harm if dist == 0 else dist for dist in dist_vec]
      agg_dist = len(dist_vec) / np.sum(1.0/np.array(dist_vec))
    else:
      raise ValueError(f'Unknown aggregation type specified {agg_type}')

    # conf. is computed as 1 - the aggregated distance
    conf_list_over_samples[i] = 1 - agg_dist
    #conf_list_over_samples[this_index] = 1 - (sum(dist_vec)/num_samples-1) # this only takes avg. distance w.r.t other points

  return conf_list_over_samples


# compute pairwise similarity between 2 SQL responses
def pairwise_sim(sql_1, sql_2, sim_type, sbert_model):

  if sim_type == 'jaccard':
    sim = jaccard_sim(sql_1, sql_2)
  elif 'rouge' in sim_type:
    sim = rouge_sim(sql_1, sql_2, sim_type)
  elif sim_type == 'sbert':
    sim = sbert_sim(sql_1, sql_2, sbert_model)
  elif sim_type == 'output_type':
    sim = output_type_sim(sql_1, sql_2)
  elif sim_type == 'aligon':
    sim = aligon_sim(sql_1, sql_2)
  else:
    raise ValueError(f'Unknown similarity type specified {sim_type}')

  return sim


# compute jaccard similarity b/w SQL queries
def jaccard_sim(sql_1, sql_2):

  #print(sql_1)
  #print(sql_2)

  list1 = sql_1.split() if sql_1 is not None else []
  list2 = sql_2.split() if sql_2 is not None else []

  #sim = metrics.jaccard_score(toks_1, toks_2)
  #intersection = len(list(set(list1).intersection(list2)))
  #union = (len(list1) + len(list2)) - intersection
  # make similarity 0 when there is an issue splitting the sqls
  #sim = 0 if len(list1) == 0 and len(list2) == 0 else float(intersection) / union

  return jaccard_sim_over_lists(list1, list2)


# compute jaccard for generic lists
def jaccard_sim_over_lists(list1, list2):

  intersection = len(list(set(list1).intersection(list2)))
  union = (len(list1) + len(list2)) - intersection
  # make similarity 0 when there is an issue splitting the sqls
  sim = 0 if len(list1) == 0 and len(list2) == 0 else float(intersection) / union

  return sim


# compute rouge similarity
# rouge type is the same as sim_type: 'rouge1', 'rouge2', 'rougeL'
def rouge_sim(sql_1, sql_2, rouge_type):

  results = scorer.score(sql_1, sql_2) # using rouge_score
  if rouge_type in results:
    sim = results[rouge_type].fmeasure
    #print('sim:', sim)
  else:
    raise ValueError(f'Unknown rouge type specified {rouge_type}')

  return sim


# compute sBERT cosine similarity
def sbert_sim(sql_1, sql_2, sbert_model):

  #sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
  sqls = [sql_1, sql_2]
  sql_embeddings = sbert_model.encode(sqls)
  similarity_score = metrics.pairwise.cosine_similarity([sql_embeddings[0]], [sql_embeddings[1]])

  return similarity_score[0][0]


# function tokenizes SQL query
def tokenize_SQL(sql):
 query_toks = wordpunct_tokenize(sql)

 return query_toks


# function identifies the SQL output type of the provided SQL using the query tokens
def check_output_type(query_toks):
  # first check if nested
  if 'SELECT' in query_toks and '(' in query_toks:
    open_brack_bool = False  # whether an open bracket is currently still open
    # loop over tokens
    for tok in query_toks:
      if tok == 'SELECT' or tok == 'select':
        # it is nested if a select is seen while a bracket has not yet been closed
        if open_brack_bool:
          output_type = 'nested'
          return output_type
      elif tok == '(':
        open_brack_bool = True
      elif tok == ')':
        open_brack_bool = False
  # check for join
  if 'JOIN' in query_toks or 'join' in query_toks:
    output_type = 'join'
  else:
    output_type = 'simple'

  return output_type


# compute output type similarity
def output_type_sim(sql_1, sql_2):

  output_type_1 = check_output_type(tokenize_SQL(sql_1))
  output_type_2 = check_output_type(tokenize_SQL(sql_2))
  sim = 1 if output_type_1 == output_type_2 else 0

  return sim


# compute aligon similarity -- need to fix this later to address parsing problems
def aligon_sim(sql_1, sql_2):
  #print('sql_1:', sql_1)
  #print('sql_2:', sql_2)

  weights = np.array([1/3, 1/3, 1/3])

  # if 'select' not in sql_1 or not matched_brackets(sql_1) or not contains_list(sql_1):
  #   select_1, where_1, group_by_1 = [], [], []
  # else:
  #   try:
  #     parsed_sql_1 = Parser(sql_1).columns_dict
  #     select_1 = parsed_sql_1['select'] if 'select' in parsed_sql_1 else []
  #     where_1 = parsed_sql_1['where'] if 'where' in parsed_sql_1 else []
  #     group_by_1 = parsed_sql_1['group_by'] if 'group_by' in parsed_sql_1 else []
  #   except:
  #     print('Could not parse!')
  #     select_1, where_1, group_by_1 = [], [], []

  try:
    parsed_sql_1 = Parser(sql_1).columns_dict

    select_1 = parsed_sql_1['select'] if 'select' in parsed_sql_1 else []
    if contains_list(select_1):
      select_1 = []
    where_1 = parsed_sql_1['where'] if 'where' in parsed_sql_1 else []
    if contains_list(where_1):
      where_1 = []
    group_by_1 = parsed_sql_1['group_by'] if 'group_by' in parsed_sql_1 else []
    if contains_list(group_by_1):
      group_by_1 = []
    else:
      select_1, where_1, group_by_1 = [], [], []
  except:
    print('Could not parse SQL for Aligon!')
    select_1, where_1, group_by_1 = [], [], []

  # if 'select' not in sql_2 or not matched_brackets(sql_2) or not contains_list(sql_2):
  #   select_2, where_2, group_by_2 = [], [], []
  # else:
  #   try:
  #     parsed_sql_2 = Parser(sql_2).columns_dict
  #     select_2 = parsed_sql_2['select'] if 'select' in parsed_sql_2 else []
  #     where_2 = parsed_sql_2['where'] if 'where' in parsed_sql_2 else []
  #     group_by_2 = parsed_sql_2['group_by'] if 'group_by' in parsed_sql_2 else []
  #   except:
  #     print('Could not parse!')
  #     select_2, where_2, group_by_2 = [], [], []

  try:
    parsed_sql_2 = Parser(sql_2).columns_dict

    select_2 = parsed_sql_2['select'] if 'select' in parsed_sql_2 else []
    if contains_list(select_2):
      select_2 = []
    where_2 = parsed_sql_2['where'] if 'where' in parsed_sql_2 else []
    if contains_list(where_2):
      where_2 = []
    group_by_2 = parsed_sql_2['group_by'] if 'group_by' in parsed_sql_2 else []
    if contains_list(group_by_2):
      group_by_2 = []

  except:
    print('Could not parse SQL for Aligon!')
    select_2, where_2, group_by_2 = [], [], []

  #print('select_1:', select_1)
  #print('select_2:', select_2)

  sim_select = jaccard_sim_over_lists(select_1, select_2)
  sim_where = jaccard_sim_over_lists(where_1, where_2)
  sim_group_by = jaccard_sim_over_lists(group_by_1, group_by_2)

  sims = np.array([sim_select, sim_where, sim_group_by])

  return np.dot(weights, sims)


# function checks if there are matched brackets in a string
def matched_brackets(str):
  count = 0
  for i in str:
    if i == "(":
      count += 1
    elif i == ")":
      count -= 1
    if count < 0:
      return False
  return count == 0


# function checks if list contains list
def contains_list(list):
  for i in list:
    if type(i) == list:
      return True
  return False


# formulate similarity matrix (aka weighted adjacency matrix) W from similarities
def get_W(gen_sql_list, sim_type, sbert_model=None):

  if sim_type == 'sbert': # hardcoded the type of sentence BERT model
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

  num_samples = len(gen_sql_list)
  W = np.zeros((num_samples, num_samples))

  for i in range(num_samples):
    for j in range(num_samples):
      if i < j:
        W[i, j] = pairwise_sim(gen_sql_list[i], gen_sql_list[j], sim_type, sbert_model)
      elif i > j:
        W[i, j] = W[j, i]
      else: # similarity = 1 when i = j !!!
        W[i, j] = 1

  #for index1 in range(num_samples):
  #  this_gen_sql = gen_sql_list[index1]
  #  W[index1, :] = [0 if index2 == index1 else pairwise_sim(this_gen_sql, gen_sql_list[index2], sim_type)
  #                  for index2 in range(num_samples)]

  return W


# function prepares a similarity dictionary from pre-computed pairwise similarities
def prepare_sim_dict(sim_type):

  if sim_type != 'sbert':
    print('This only works for sbert!')
    return

  data_path_string = "pairwise_sim_metric_data/"
  file = PRECOMPUTED_SIMS_CASE + '_' + sim_type + '.json'

  # load file
  INPUT_FILE = os.path.abspath(data_path_string + file)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    sim_metric_data = json.load(f)

  # prepare dict where keys are tuples
  sim_dict = {}
  for question in sim_metric_data:
    if question in sim_metric_data:
      this_sim_data = sim_metric_data[question]
    else:
      raise ValueError('This question is not present in the precomputed file!')
    this_pair_dict = {}
    for pair_info in this_sim_data:
      this_pair_dict[(pair_info['sql_1'], pair_info['sql_2'])] = pair_info['sim']
    sim_dict[question] = this_pair_dict

  return sim_dict


# function prepares a similarity dictionary from pre-computed pairwise similarities for ettubench

def prep_and_save_sim_dict_ettubench(dataset, sim_type, save_bool=False):

  data_path_string = "pairwise_sim_metric_data/"
  if dataset == 'spider_codellama':
    file = "sim_dict_spider_dev_codellama_temp_all_ettubench.json"
  elif dataset == 'spider_realistic_codellama':
    file = "sim_dict_spider_realistic_dev_codellama_temp_all_ettubench.json"
  else:
    raise ValueError('This dataset is not allowed for sim dict preparation!')

  # load file
  INPUT_FILE = os.path.abspath(data_path_string + file)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    data = json.load(f)

  # prepare dict where keys are tuples
  sim_dict = {}
  for question in data:
    sim_dict[question] = {}
    for pair in data[question]:
      this_pair_info = data[question][pair]
      sql_1 = this_pair_info['sql_1']
      sql_2 = this_pair_info['sql_2']
      if (sql_1, sql_2) not in sim_dict[question]: # and (sql_2, sql_1) not in sim_dict[question]:
        sim_dict[question][(sql_1, sql_2)] = this_pair_info[sim_type]

        # if sql_1 == sql_2:
        #   sim_dict[question][(sql_1, sql_2)] = 1.0
        # else:
        #   if sim_type in this_pair_info:
        #     sim_dict[question][(sql_1, sql_2)] = this_pair_info[sim_type]
        #   else:
        #     print('This sim type is not present in this entry!')

  # save file
  output_filename = 'sim_dict_' + dataset + '_' + sim_type + '.json'

  if save_bool:
    with open(output_filename, "w") as outfile:
      json.dump(sim_dict, outfile)

  return sim_dict


# function prepares similarity matrix from precomputed similarities
def get_W_from_precomputed_sims(gen_sql_list, question, sim_dict):

  default_sim = 0 # this is the default similarity when a pair is not found in sim_dict

  if question in sim_dict:
    sim_info_this_query = sim_dict[question]
  else:
    raise ValueError('This question is not present in the similarity dictionary!')

  num_samples = len(gen_sql_list)
  W = np.zeros((num_samples, num_samples))

  for i in range(num_samples):
    for j in range(num_samples):
      sql_i = gen_sql_list[i]
      sql_j = gen_sql_list[j]
      if i != j:
        if (sql_i, sql_j) in sim_info_this_query:
          W[i, j] = sim_info_this_query[(sql_i, sql_j)]
        elif (sql_j, sql_i) in sim_info_this_query:
          W[i, j] = sim_info_this_query[(sql_j, sql_i)]
        else:
          #print(question)
          #print(sql_i)
          #print(sql_j)
          W[i, j] = default_sim
          #raise ValueError('There is a missing sql pair in the similarity dictionary!')

      else:  # similarity = 1 when i = j !!!
        W[i, j] = 1

  return W



################################
# SPECTRAL CLUSTERING
################################


# compute the degree matrix from the weighted adjacency matrix
def get_D_mat(W):

  D = np.diag(np.sum(W, axis=1))
  return D


# compute the normalized Laplacian matrix from the degree matrix and weighted adjacency matrix
def get_L_mat(W, symmetric=True):
  # compute the degreee matrix from the weighted adjacency matrix
  D = np.diag(np.sum(W, axis=1))
  if symmetric:
    L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
  else:
    raise NotImplementedError()
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    #L = np.linalg.inv(D) @ (D - W)
  return L.copy()


def get_eig(L, thres=None, eps=None):
  # This function assumes L is symmetric
  # compute the eigenvalues and eigenvectors of the Laplacian matrix
  if eps is not None:
    L = (1 - eps) * L + eps * np.eye(len(L))
  eigvals, eigvecs = np.linalg.eigh(L)

  if thres is not None:
    keep_mask = eigvals < thres
    eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
  return eigvals, eigvecs


# get confidence from degree, or eigen vectors
def get_conf_spectral_clustering(W, spec_metric='degree', thres=0.9, eps=None):

  num_samples = len(W[0, :])

  if spec_metric == 'degree':
    # compute the degree matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    conf_list_over_samples = [(D[i, i] / num_samples) for i in range(num_samples)]
    return conf_list_over_samples

  if spec_metric == 'ecc':
    L = get_L_mat(W)
    # find eigen vectors
    eigvals, eigvecs = get_eig(L, thres, eps)
    # apply threshold to keep only selected eigen vectors
    keep_mask = eigvals < thres
    eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    ds = np.linalg.norm(eigvecs - eigvecs.mean(0)[None, :], 2, axis=1)
    conf_list_over_samples = list(ds)
    return conf_list_over_samples

  raise ValueError(f'Unknown metric type specified for spectral clustering approach {spec_metric}')


################################
# BAYESIAN SIMILARITY
################################


# compute confidence estimates using a Bayesian approach from the similarity matrix
def conf_estimation_bayes_single_query(W, bayes_type, bayes_param_dict, eps):

  # bayes type is like post:mixed or prior:beta or post_equi:mixed

  bayes_setting = bayes_type.split(':')[0]
  sim_dist_type = bayes_type.split(':')[1]

  num_samples = len(W[0, :])

  if bayes_setting == 'prior':
    p = bayes_param_dict['p']

  else:

    # retrieve parameters needed for the probability computation
    if sim_dist_type == 'beta' or sim_dist_type == 'beta_mix':

      (p, weights_C, alphas_C, betas_C, weights_I, alphas_I, betas_I) = \
        (bayes_param_dict['p'], bayes_param_dict['weights_C'], bayes_param_dict['alphas_C'],
         bayes_param_dict['betas_C'], bayes_param_dict['weights_I'], bayes_param_dict['alphas_I'],
         bayes_param_dict['betas_I'])

    elif sim_dist_type == 'mixed':

      (p, p_0_C, p_1_C, alpha_C, beta_C, p_0_I, p_1_I, alpha_I, beta_I) = \
        (bayes_param_dict['p'], bayes_param_dict['p_0_C'], bayes_param_dict['p_1_C'], bayes_param_dict['alpha_C'],
         bayes_param_dict['beta_C'], bayes_param_dict['p_0_I'], bayes_param_dict['p_1_I'], bayes_param_dict['alpha_I'],
         bayes_param_dict['beta_I'])

    else:
      raise ValueError(f'Unknown similarity distribution type specified {sim_dist_type}')


  # (p, alpha_C, beta_C, alpha_I, beta_I) = (bayes_param_dict['p'], bayes_param_dict['alpha_C'],
  #                                            bayes_param_dict['beta_C'], bayes_param_dict['alpha_I'],
  #                                            bayes_param_dict['beta_I'])

  conf_list = []
  for i in range(num_samples):

    if bayes_setting == 'prior':
      prob = p
    else:

      data_this_query = list(np.delete(W[i, :], i))  # take i^th row of similarity matrix and remove i^th index (as sim = 0)
      data_this_query = [1 - eps if x >= 1 else eps if x <= 0 else x for x in data_this_query]  # avoid 0s and 1s

      if sim_dist_type == 'beta' or sim_dist_type == 'beta_mix':

        pdfs_C = [compute_pdf_mixture_betas(sim, weights_C, alphas_C, betas_C) for sim in data_this_query]
        pdfs_I = [compute_pdf_mixture_betas(sim, weights_I, alphas_I, betas_I) for sim in data_this_query]

      else:

        pdfs_C = [compute_pdf_mixed(sim, p_0_C, p_1_C, alpha_C, beta_C, eps) for sim in data_this_query]
        pdfs_I = [compute_pdf_mixed(sim, p_0_I, p_1_I, alpha_I, beta_I, eps) for sim in data_this_query]

      if bayes_setting == 'post':
        den = (p * np.prod(pdfs_C)) + ((1 - p) * np.prod(pdfs_I))
        if den <= 0:
          prob = p  # set to prior in case there are computational issues with computing den
        else:
          prob = (p * np.prod(pdfs_C)) / den  # using learned priors
      elif bayes_setting == 'post_equi':
        den = np.prod(pdfs_C) + np.prod(pdfs_I)
        if den <= 0:
          prob = p
        else:
          prob = np.prod(pdfs_C) / den  # using equi-probable prior
      else:
        if bayes_setting != 'prior':
          raise ValueError(f'Unknown bayes setting specified {bayes_setting}')


    # if sim_dist_type == 'beta':
    #
    #   pdfs_C = beta.pdf(data_this_query, alpha_C, beta_C, loc=0, scale=1)
    #   pdfs_I = beta.pdf(data_this_query, alpha_I, beta_I, loc=0, scale=1)
    #


    # do some error correction if needed
    if prob < 0:
      prob = 0
    if prob > 1:
      prob = 1
    if np.isnan(prob): # this is an issue for sim_type = output_type
      #print('Nan prob!')
      prob = p # make prior

    conf_list.append(prob)

  return conf_list



# compute pdf of mixture of betas for similarity (sim) where the data is in the following form
# {
#     'weights': [w_1, w_2, ...],
#     'alphas': [alpha_1, alpha_2, ...],
#     'betas': [alpha_1, alpha_2, ...]
#   }
def compute_pdf_mixture_betas(sim, weights, alphas, betas):

  likelihood_per_comp = [weights[i] * beta.pdf(sim, alphas[i], betas[i], loc=0, scale=1) for i in range(len(weights))]

  return sum(likelihood_per_comp)



# function prepares bayes param dict
def prepare_bayes_param_dict(split_index, uq_type, sim_dist_type, beta_data_given_correct,
                             beta_data_given_incorrect, eps):

  if 'post' in uq_type:
    if sim_dist_type == 'beta':
      alpha_C, beta_C, alpha_I, beta_I = \
        fit_sim_dist(beta_data_given_correct, beta_data_given_incorrect)
      bayes_param_dict = {
        'weights_C': [1],
        'alphas_C': [alpha_C],
        'betas_C': [beta_C],
        'weights_I': [1],
        'alphas_I': [alpha_I],
        'betas_I': [beta_I]
      }
    elif sim_dist_type == 'mixed':
      p_0_C, p_1_C, alpha_C, beta_C, p_0_I, p_1_I, alpha_I, beta_I = \
        fit_sim_dist_mixed(beta_data_given_correct, beta_data_given_incorrect, eps)
      bayes_param_dict = {
        'p_0_C': p_0_C,
        'p_1_C': p_1_C,
        'alpha_C': alpha_C,
        'beta_C': beta_C,
        'p_0_I': p_0_I,
        'p_1_I': p_1_I,
        'alpha_I': alpha_I,
        'beta_I': beta_I
      }
    elif sim_dist_type == 'beta_mix':  # harcoded for now
      # bayes_param_dict = uq_from_similarity.beta_mixture_params_codellama()[split_index]
      bayes_param_dict = uq_from_similarity.beta_mixture_params_codellama_v2()[split_index]
      # bayes_param_dict = uq_from_similarity.beta_mixture_params_deepseek()[split_index]
    else:
      raise ValueError(f'Invalid similarity distribution type {sim_dist_type}')
  else:
    bayes_param_dict = {}

  return bayes_param_dict



# function prepares the data needed for Bayesian methods
# eps is some epsilon margin to avoid value of 0 or 1, for beta fitting
def prepare_bayes_data(samples_data_val, sim_type, sim_dict, eps):

  all_beta_data = []
  beta_data_given_correct = []
  beta_data_given_incorrect = []
  acc_list_full = []

  # compile the data for learning parameters
  for query in samples_data_val:
    samples = query['samples']
    num_samples = len(samples)

    gen_sql_list = [sample['gen_sql'] for sample in samples]
    acc_list = [sample['exec_acc'] for sample in samples]

    acc_list_full += acc_list

    # formulate similarity matrix -- if possible, prepare it from precomputed similarities
    if sim_type in PRECOMPUTED_SIMS:
      question = query['question']
      W = get_W_from_precomputed_sims(gen_sql_list, question, sim_dict)
    else:
      W = get_W(gen_sql_list, sim_type)  # -- SBERT model is hardcoded in this function

    for i in range(num_samples):
      data_this_query = list(np.delete(W[i,:], i)) # take i^th row of similarity matrix and remove i^th index (as sim = 0)
      data_this_query = [1 - eps if x >= 1 else eps if x <= 0 else x for x in data_this_query] # avoiding 0s and 1s

      all_beta_data += data_this_query

      if acc_list[i] == 1:
        # add similarity w.r.t to all other points
        beta_data_given_correct += data_this_query
      else:
        beta_data_given_incorrect += data_this_query

  return acc_list_full, beta_data_given_correct, beta_data_given_incorrect, all_beta_data


def fit_sim_dist(beta_data_given_correct, beta_data_given_incorrect):

  # learn parameters for the 2 Beta distributions

  alpha_C, beta_C = fit_beta(beta_data_given_correct)
  alpha_I, beta_I = fit_beta(beta_data_given_incorrect)

  return alpha_C, beta_C, alpha_I, beta_I


# fit mixture distribution -- masses near 0 and 1 and beta for the rest
def fit_sim_dist_mixed(beta_data_given_correct, beta_data_given_incorrect, eps):

  # learn parameters for distribution conditioned on correct response
  p_0_C = sum(np.array(beta_data_given_correct) == eps) / len(beta_data_given_correct)
  p_1_C = sum(np.array(beta_data_given_correct) == 1 - eps)/ len(beta_data_given_correct)
  reduced_data_correct = [sim for sim in beta_data_given_correct if sim != eps and sim != (1 - eps)]
  alpha_C, beta_C = fit_beta(reduced_data_correct)

  # learn parameters for the 2 Beta distributions
  p_0_I = sum(np.array(beta_data_given_incorrect) == eps) / len(beta_data_given_incorrect)
  p_1_I = sum(np.array(beta_data_given_incorrect) == 1 - eps) / len(beta_data_given_incorrect)
  reduced_data_incorrect = [sim for sim in beta_data_given_incorrect if sim != eps and sim != (1 - eps)]
  alpha_I, beta_I = fit_beta(reduced_data_incorrect)

  return p_0_C, p_1_C, alpha_C, beta_C, p_0_I, p_1_I, alpha_I, beta_I


def compute_pdf_mixed(sim, p_0, p_1, alpha_param, beta_param, eps):

  #print(sim)
  if sim == eps:
    prob = p_0
  elif sim == 1 - eps:
    prob = p_1
  else:
    prob = (1 - (p_0 + p_1)) * beta.pdf(sim, alpha_param, beta_param, loc=0, scale=1)

  return prob


def log_likelihood(data, method, correct_bool, bayes_params_dict):

  ll = 0
  if method == 'beta':
    alphas = bayes_params_dict
    #for sim in data:

  else:

    d = 0


  return



# method is either 'moments' or 'fit' -- only moments enabled for now
def fit_beta(data, method='moments'):

  # ----------------Fit using moments----------------
  mean = np.mean(data)
  var = np.var(data, ddof=1)
  alpha_est = mean ** 2 * (1 - mean) / var - mean
  beta_est = alpha_est * (1 - mean) / mean

  # ----------------Fit using beta.fit----------------
  # alpha_est, beta_est, _, _ = beta.fit(data)

  return alpha_est, beta_est



################################
# CALSSIFICATION, EX: LOGISTIC REGRESSION
################################


def conf_estimation_clf_single_query(W, clf, eps):

  num_samples = len(W[0, :])
  conf_list = []
  for i in range(num_samples):
    # prepare the features

    #data_this_query = W[i,:]
    #data_this_query = [np.mean(W[i,:])]
    #data_this_query = list(np.delete(W[i, :], i)) + [np.mean(W[i,:])]
    data_this_query = list(np.delete(W[i, :], i))

    # off_diag = []
    # for k in range(1, len(W[0, :])):
    #   off_diag += list(np.diag(W, k))
    # data_this_query = np.array(off_diag)

    data_this_query = [1 - eps if x >= 1 else eps if x <= 0 else x for x in data_this_query]

    # try sorting! --- this will create features in order of increasing similarity
    data_this_query = np.sort(data_this_query)

    data_this_query = logit(data_this_query)  # transform numbers to real line

    # predict probability from logistic regression classifier
    # we take the only element of the matrix, and then take the prob. corresponding to class of 1
    prob = clf.predict_proba([data_this_query])[0][1]
    conf_list.append(prob)

  return conf_list


# function prepares the data needed for LR method
def prepare_clf_data(samples_data_val, sim_type, sim_dict, eps):

  X = []
  acc_list_full = []

  # compile the data for learning parameters
  for query in samples_data_val:
    samples = query['samples']
    num_samples = len(samples)

    gen_sql_list = [sample['gen_sql'] for sample in samples]
    acc_list = [sample['exec_acc'] for sample in samples]
    # acc_list = [samples[0]['exec_acc']] -- if we only want to consider the first sample
    acc_list_full += acc_list

    # formulate similarity matrix -- if possible, prepare it from precomputed similarities
    if sim_type in PRECOMPUTED_SIMS:
      question = query['question']
      W = get_W_from_precomputed_sims(gen_sql_list, question, sim_dict)
    else:
      W = get_W(gen_sql_list, sim_type)  # -- SBERT model is hardcoded in this function
    #print('W:', W)

    # num_samples = 1 -- if we only want to consider the first sample
    for i in range(num_samples):
      #data_this_query = W[i,:]
      # data_this_query = [np.mean(W[i, :])]
      #data_this_query = list(np.delete(W[i, :], i)) + [np.mean(W[i,:])]
      data_this_query = list(np.delete(W[i, :], i))

      # off_diag = []
      # for k in range(1, len(W[0, :])):
      #   off_diag += list(np.diag(W, k))
      # data_this_query = np.array(off_diag)

      data_this_query = [1 - eps if x >= 1 else eps if x <= 0 else x for x in data_this_query]

      # try sorting! --- this will create features in order of increasing similarity
      data_this_query = np.sort(data_this_query)

      data_this_query = logit(data_this_query)

      X.append(data_this_query)

  return acc_list_full, X


# fit classifier for accuracy where features are similarities
# limited set of clf types are allowed
def fit_classifier(acc_list_full, X, clf_type):

  if clf_type == 'lr': # logistic regression
    weights = {0: 1, 1: 1}
    clf = LogisticRegression(random_state=0, class_weight=weights, fit_intercept=True).fit(X, acc_list_full)
  elif clf_type == 'rf': # random forest
    #max_depth = None
    max_depth = 4  # maybe choose 3?
    #max_depth = 5
    clf = RandomForestClassifier(max_depth=max_depth, random_state=0).fit(X, acc_list_full)
  elif clf_type == 'qda': # qda
    clf = QDA().fit(X, acc_list_full)
  elif clf_type == 'svm': # svm
    clf = svm.SVC(probability=True).fit(X, acc_list_full)
  else:
    raise ValueError(f'Invalid classifier type {clf_type}')

  return clf


################################
# SIM RELATED UTILS
################################

# get all pairwise sims and store them as both a list and a dict
def get_all_pairwise_sims(samples_data, sim_type):

  all_pairwise_sims = []
  all_pairwise_sims_dict = {}

  query_index = 0
  for query in samples_data:
    question = query['question']
    gen_sql_list = [sample['gen_sql'] for sample in query['samples']]
    # formulate similarity matrix
    W = get_W(gen_sql_list, sim_type) # -- SBERT model is hardcoded in this function

    # get all pairs to append to list
    off_diag = []
    for k in range(1, len(W[0, :])):
      off_diag += list(np.diag(W, k))
    all_pairwise_sims += off_diag

    #all_pairwise_sims_dict[question] = {}
    # get all pairs and add as list to dict
    list_with_sims = []
    for i in range(len(W[0, :])):
      for j in range(len(W[0, :])):
        if i != j:
          list_with_sims.append({'sql_1': gen_sql_list[i], 'sql_2': gen_sql_list[j], 'sim': W[i,j]})
    all_pairwise_sims_dict[question] = list_with_sims

    # evaluating rouge and sbert takes time, so keep track of confidence estimation for queries by printing
    batch_size = 50
    if 'rouge' in sim_type or sim_type == 'sbert':
      if (query_index + 1) % batch_size == 0:
        print('Obtained similarity pairs up to query:', query_index + 1)

    query_index += 1

  return all_pairwise_sims, all_pairwise_sims_dict




def beta_mixture_params_codellama_v1():

  param_dict = {
    1: {
      'weights_C': [0.4395961, 0.5604039],
      'alphas_C': [2.8315284,	0.27559],
      'betas_C': [2.7327448, 0.1800655],
      'weights_I': [0.8950685, 0.1049315],
      'alphas_I': [0.5069122, 0.2512545],
      'betas_I': [1.4031706, 0.1602399]
    },
    2: {
      'weights_C': [0.438582,	0.561418],
      'alphas_C': [2.5610178,	0.2799961],
      'betas_C': [2.6941168, 0.17837],
      'weights_I': [0.1119918, 0.8880082],
      'alphas_I': [0.2234121, 0.530328],
      'betas_I': [0.160249, 1.4876386]
    },
    3: {
      'weights_C': [0.4480018, 0.5519982],
      'alphas_C': [2.8512839, 0.2772629],
      'betas_C': [3.1644838, 0.1808422],
      'weights_I': [0.8712454, 0.1287546],
      'alphas_I': [0.5293171, 0.2217805],
      'betas_I': [1.437309, 0.1622109]
    },
    4: {
      'weights_C': [0.4676535, 0.5323465],
      'alphas_C': [2.6985221,	0.260076],
      'betas_C': [2.7662304, 0.179675],
      'weights_I': [0.8669432, 0.1330568],
      'alphas_I': [0.5178852, 0.2308726],
      'betas_I': [1.4156016, 0.1607283]
    },
    5: {
      'weights_C': [0.4542799, 0.5457201],
      'alphas_C': [2.607761, 0.2622566],
      'betas_C': [2.6769116, 0.1811766],
      'weights_I': [0.873005, 0.126995],
      'alphas_I': [0.5391712, 0.2290828],
      'betas_I': [1.4385565, 0.162249]
    }
  }

  return param_dict



def beta_mixture_params_codellama_v2():

  param_dict = {
    1: {
      'weights_C': [0.4866264,	0.3378758,	0.1754978],
      'alphas_C': [2.198058,	29714.00451,	34.606843],
      'betas_C': [2.423425,	3.458169,	7.398736],
      'weights_I': [0.1346675,	0.4262031,	0.4391294],
      'alphas_I': [0.5695008,	2.7935223,	1.9637309],
      'betas_I': [0.1595145,	2.2880544,	2.3433691]
    },
    2: {
      'weights_C': [0.3052,	0.4922899,	0.2025101],
      'alphas_C': [41125.76995,	2.743384,	1.445598],
      'betas_C': [4.602922,	1.835293,	1.777648],
      'weights_I': [0.4248692,	0.1358639,	0.4392669],
      'alphas_I': [2.1057198,	0.5614402,	2.712535],
      'betas_I': [2.4210965,	0.1572131,	2.3869492]
    },
    3: {
      'weights_C': [0.4984579,	0.3232322,	0.1783099],
      'alphas_C': [2.775124,	43143.21628,	1.400683],
      'betas_C': [2.046449,	4.805253,	1.814953],
      'weights_I': [0.1573634,	0.4150299,	0.4276066],
      'alphas_I': [0.5014406,	2.469361,	2.4693623],
      'betas_I': [0.155212,	2.4718127,	2.4718126]
    },
    4: {
      'weights_C': [0.5186894,	0.2823956,	0.198915],
      'alphas_C': [2.641801,	42785.57975,	30.906609],
      'betas_C': [3.066381,	4.769489,	7.123134],
      'weights_I': [0.4128901,	0.088,	0.4991099],
      'alphas_I': [5.697462,	31858.94968,	2.474406],
      'betas_I': [2.755199,	3.67293,	4.390616]
    },
    5: {
      'weights_C': [0.4673091,	0.2291162,	0.3035747],
      'alphas_C': [2.666396,	25.545327,	72851.52746],
      'betas_C': [3.332946,	6.369347,	7.779696],
      'weights_I': [0.4118269,	0.1638666,	0.4243065],
      'alphas_I': [2.3207546,	0.5966693,	2.3207546],
      'betas_I': [2.3104609,	0.1580994,	2.3104609]
    }
  }

  return param_dict



def beta_mixture_params_deepseek():

  param_dict = {
    1: {
      'weights_C': [0.011, 0.989],
      'alphas_C': [16388.36, 3.90],
      'betas_C': [2.11, 2.50],
      'weights_I': [0.986, 0.014],
      'alphas_I': [3.62, 19805.44],
      'betas_I': [2.48, 2.46]
    } #,
    #2: x,
    #3: x,
    #4: x,
    #5: x
  }

  return param_dict


