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
import uq_from_similarity # file with methods for similarity and confidence estimation

# import scripts that are needed
import gen_calibration_error
#import visualizer

from sklearn import metrics
#from sklearn.linear_model import LogisticRegression

from nltk.tokenize import wordpunct_tokenize # used for tokenization for SQL type

from rouge_score import rouge_scorer # used for computing rouge score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from sql_metadata import Parser # used for computing Aligon score

from sentence_transformers import SentenceTransformer # used for computing sBERT
SBERT_MODEL_NAME = "all-MiniLM-L12-v2" # this is supposedly 5 times faster than "all-mpnet-base-v2"
#SBERT_MODEL_NAME = "all-mpnet-base-v2" # this takes a while
#SBERT_MODEL_NAME = "bert-base-nli-mean-tokens" # this does not seem to be faster than "all-MiniLM-L12-v2"
#SBERT_MODEL_NAME = "average_word_embeddings_glove.840B.300d" # non-transformer embedding; this seems slow

PRECOMPUTED_SIMS = ['sbert', 'aligon', 'aouiche', 'makiyama']

from scipy.stats import beta

#from sklearn.utils import check_random_state



################################
# UTILS
################################


# function removes samples as needed
def remove_samples_from_data(samples_data, restrict_sample_indices_bool = False, accepted_sample_indices = None):

  if restrict_sample_indices_bool and accepted_sample_indices is not None:
    new_samples_data = samples_data.copy()
    for s in new_samples_data:
      samples = s['samples']
      s['samples'] = [samples[i] for i in accepted_sample_indices]
      #s['samples'] = samples[0:num_accepted_sample_indices]
    return new_samples_data
  else:
    return samples_data


# function loads data
def load_samples_data_MDE55(temperature):

  # load MDE samples
  data_path_string = "samples_data/"
  file = "MDE55_samples_temp_" + str(temperature) + ".json"

  INPUT_FILE = os.path.abspath(data_path_string + file)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    data = json.load(f)

  return data


# function loads and processes Spider data from M. Glass
def load_and_process_samples_data_spider_deepseeker(sampling_type):

  # load raw data
  data_path_string = "samples_data/"

  # file = "sql_gen_all_lora_dev_deepseek_sql_exec.jsonl"
  # INPUT_FILE = os.path.abspath(data_path_string + file)
  # with open(INPUT_FILE) as f:
  #   data = [json.loads(line) for line in f]
  #
  # # process data
  # samples_data = []
  # for query in data:
  #   samples = [{'gen_sql': query['sqls'][sample_index], 'exec_acc': 1
  #   if query['execution_results'][sample_index] == 'Correct' else 0}
  #              for sample_index in range(len(query['sqls']))]
  #   processed_query = {
  #     'question': query['question'],
  #     'samples': samples
  #   }
  #   samples_data.append(processed_query)

  valid_temps = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # hard-coding the acceptable temps
  valid_temps_strings = [str(t) for t in valid_temps]

  perf_string = 'execution_match'

  file = "spider_dev_deepseek_v2.json"

  INPUT_FILE = os.path.abspath(data_path_string + file)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    data = json.load(f)

  # process data
  samples_data = []

  for question in data:
    query = data[question]
    results = query['results']
    samples = []

    if sampling_type == 'temp_first':

      for temp in results:
        if temp in valid_temps_strings:
          #print('temp:', temp)
          this_sample = {
            'gen_sql': results[temp][0]['processed_generated_sql'],
            'exec_acc': results[temp][0][perf_string] if perf_string in results[temp][0] else 0,
            'score': results[temp][0]['avg_log_prob']
          }
          samples.append(this_sample)

    # consider all generations across all temperatures -- this is a hybrid of standard and temperature sampling
    elif sampling_type == 'temp_all':
      for temp in results:
        if temp in valid_temps_strings:
          #print('temp:', temp)
          for gen_query in results[temp]:
            this_sample = {
              'gen_sql': gen_query['processed_generated_sql'],
              'exec_acc': gen_query[perf_string] if perf_string in gen_query else 0,
              'score': gen_query['avg_log_prob']
            }
            samples.append(this_sample)

    else:
      raise ValueError(f'Unknown sampling type specified {sampling_type}')

    processed_query = {
      'question': question,
      'gt_sql': query['ground_truth'],
      'samples': samples
    }
    samples_data.append(processed_query)

  return samples_data


# function loads and processes Bird data from M. Glass
def load_and_process_samples_data_bird():

  # load raw data
  data_path_string = "samples_data/"

  file_numbers = list(range(4))
  files = ['bird_dev_granite_file' + str(f_num) + '.jsonl' for f_num in file_numbers]

  samples_data = []
  for file in files:

    # load data
    INPUT_FILE = os.path.abspath(data_path_string + file)
    with open(INPUT_FILE) as f:
      data = [json.loads(line) for line in f]

    # process data
    for query in data:
      samples = []
      gt_sql = query['target']
      for sample_index in range(len(query['samples'])):
        gen_sql = query['samples'][sample_index]['text']
        # include generated sequel, reward,
        sample = {
          'gen_sql': gen_sql,
          'exec_acc': 1 if exact_match(gt_sql, gen_sql) else 0,
          'score': query['samples'][sample_index]['score']
        }
        samples.append(sample)

      processed_query = {
        'question': query['question'],
        'gt_sql': gt_sql,
        'samples': samples
      }
      samples_data.append(processed_query)


  return samples_data


# function loads and processes data for standard and temperature sampling
# sampling_type is 'temp_first' (only take first sample for each temp) or 'temp_all' (take all samples across temps)
# or 'standard', which is at a given temperature and therefore also requires a temperature to be provided
# data_model is a combination of dataset (spider) and model (codellama or granite)
# split is either 'dev' or 'test'
def load_and_process_samples_data_spider_codellama_fewshot(data_model, sampling_type, split, temperature=None):

  valid_temps = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5] # hard-coding the 6 acceptable temps
  valid_temps_strings = [str(t) for t in valid_temps]

  # performance
  perf_string = "execution_match"
  #perf_string = "exact_match"

  # load raw data
  data_path_string = "samples_data/"
  file = None

  if data_model == 'spider_codellama':
    if split == 'test':
      file = "spider_test_codellama_v1.json"
    elif split == 'dev':
      #file = "spider_dev_codellama_v1.json"
      #file = "spider_dev_codellama_v2.json"
      file = "spider_dev_codellama_v3.json"  # latest one
      #file = "spider_dev_codellama_v4.json" # --- one such that ettubench is able to parse

    else:
      raise ValueError(f'Unknown split type specified {split}')
  elif data_model == 'spider_granite':
    file = "spider_dev_granite.json"
  elif data_model == 'spider_realistic_codellama':
    if split == 'test':
      raise ValueError('Test set not available!')
    elif split == 'dev':
      file = "spider_realistic_dev_codellama.json"

  else:
    raise ValueError(f'Unknown data model type specified {data_model}')


  INPUT_FILE = os.path.abspath(data_path_string + file)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    data = json.load(f)

  # process data
  samples_data = []
  for question in data:
    query = data[question]
    results = query['results']

    samples = []

    # only consider first sample for each temperature -- this is vanilla temperature sampling
    if sampling_type == 'temp_first':

      for temp in results:
        if temp in valid_temps_strings:
          #print('temp:', temp)
          this_sample = {
            'gen_sql': results[temp][0]['processed_generated_sql'],
            'exec_acc': results[temp][0][perf_string],
            'score': results[temp][0]['avg_log_prob'] if 'avg_log_prob' in results[temp][0] else None
          }
          samples.append(this_sample)

    # consider all generations across all temperatures -- this is a hybrid of standard and temperature sampling
    elif sampling_type == 'temp_all':
      for temp in results:
        if temp in valid_temps_strings:
          #print('temp:', temp)
          for gen_query in results[temp]:
            this_sample = {
              'gen_sql': gen_query['processed_generated_sql'],
              'exec_acc': gen_query[perf_string],
              'score': gen_query['avg_log_prob'] if 'avg_log_prob' in gen_query else None
            }
            samples.append(this_sample)

    elif sampling_type == 'standard':
      if temperature in results:
        for gen_query in results[temperature]:
          this_sample = {
            'gen_sql': gen_query['processed_generated_sql'],
            'exec_acc': gen_query[perf_string],
            'score': gen_query['avg_log_prob'] if 'avg_log_prob' in gen_query else None
          }
          samples.append(this_sample)
      else:
        raise ValueError(f'The following temp is not present in the data: {temperature}')

    else:
      raise ValueError(f'Unknown sampling type specified {sampling_type}')

    processed_query = {
      'question': question,
      'gt_sql': query['ground_truth'],
      'samples': samples
    }
    samples_data.append(processed_query)

  return samples_data


# function loads and processes Spider data for standard or temperature-all sampling
def load_and_process_samples_data_spider_csv(data_model):

  # read csv data
  data_path_string = "samples_data/"

  if data_model == 'spider_fp':
    file = 'spider_dev_multigeneration_8_codellamacodellama-34b-instruct.csv'
  elif data_model == 'spider_fine-tuned_granite':
    file = 'spider_dev_fine-tuned_granite.csv'
  else:
    raise ValueError(f'Unknown data model type specified {data_model}')


  INPUT_FILE = os.path.abspath(data_path_string + file)
  rows = []
  with open(INPUT_FILE, 'r', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      rows.append(row)

  # convert rows to a dictionary
  samples_data_dict = {}

  for row in rows:
    nl_query = row['utterance']
    if nl_query == 'False': # skip the ones where NL query is 'False'
      continue

    # if this nl query not present in the dict, then create a new key-value pair
    if nl_query not in samples_data_dict:

      if data_model == 'spider_fine-tuned_granite': # some bug in how exec. accuracy was recorded
        samples = [{'gen_sql': row['pred_sql'], 'exec_acc': 1 if row['correct_applies_applies'] == 'True' else 0,
                   'score': float(row['prob_score'])}]
      else:
        samples = [{'gen_sql': row['pred_sql'], 'exec_acc': 1 if row['correct_applies'] == 'TRUE' else 0}]
      query = {
        # 'question': nl_query,
        'gt_sql': row['gold_sql'],
        'samples': samples
      }
      samples_data_dict[nl_query] = query
    else: # append the new sample in the existing entry for this nl query
      samples = samples_data_dict[nl_query]['samples']
      if data_model == 'spider_fine-tuned_granite':  # some bug in how exec. accuracy was recorded
        this_sample = {'gen_sql': row['pred_sql'], 'exec_acc': 1 if row['correct_applies_applies'] == 'True' else 0,
                       'score': float(row['prob_score'])}
      else:
        this_sample = {'gen_sql': row['pred_sql'], 'exec_acc': 1 if row['correct_applies'] == 'TRUE' else 0}
      samples.append(this_sample)
      samples_data_dict[nl_query]['samples'] = samples

  # convert dictionary to list
  samples_data = [{'nl_query': nl_query, 'gt_sql': samples_data_dict[nl_query]['gt_sql'],
                   'samples': samples_data_dict[nl_query]['samples']} for nl_query in samples_data_dict]

  # remove instances that do not have the same number of samples as the first query
  num_samples = len(samples_data[0]['samples'])
  include = [len(query['samples'])==num_samples for query in samples_data]
  samples_data = [c for c, i in zip(samples_data, include) if i]

  return samples_data


# compute whether there is exact match between strings
def exact_match(str1, str2):
  return str1.strip().upper() == str2.strip().upper()



################################
# EVAL
################################


def evaluate_metrics(accuracy_list, conf_list, num_bins):

  fpr, tpr, thresholds = metrics.roc_curve(accuracy_list, conf_list, pos_label=1)
  auroc = metrics.auc(fpr, tpr)
  auarc = compute_auarc(accuracy_list, conf_list)
  ece = \
    gen_calibration_error.ece(labels=accuracy_list, probs=conf_list, num_bins=num_bins)
  ace = \
    gen_calibration_error.ace(labels=accuracy_list, probs=conf_list, num_bins=num_bins)
  rmsce = \
    gen_calibration_error.rmsce(labels=accuracy_list, probs=conf_list, num_bins=num_bins)

  return auroc, auarc, ece, ace, rmsce


# perform evaluation of confidence from generated samples
# format for generation/sampling based approaches: gens|sim_type|uq_from_sim_type
# other
# sim_type: output_type, jaccard, rouge1, rouge2, rougeL, sbert, ADD MORE SQL specific ones
# uq_from_sim_type: agg-arith, agg-geom, agg-harm (simple similarity aggregation), spec-degree (spectral clustering with degree),
# spec-ecc (eccentricity - U version in Lin et al. paper), bayes-post, bayes-prior, bayes-post_equi (bayesian aggregation)
def eval_samples(split_index, samples_data_valid, samples_data_test, setting, num_bins = 10,
                 restrict_sample_indices_bool = False, accepted_sample_indices = None):

  eps = 0.0001 # hardcoded for now

  # valid list of sim_types, i.e. those that are enabled
  valid_sim_types = ['output_type', 'jaccard', 'rouge1', 'rouge2', 'rougeL', 'sbert', 'aligon', 'aouiche', 'makiyama']

  bayes_param_dict = {}
  clf = None

  num_queries = len(samples_data_test)
  num_samples = len(samples_data_test[0]['samples'])
  exec_acc_vec, auroc_vec, auarc_vec, ece_vec, ace_vec, rmsce_vec = [], [], [], [], [], []

  # for similarity-based approaches, compute a dictionary with confidences
  if 'gens' in setting:
    sim_type, uq_type = setting.split('|')[1], setting.split('|')[2]

    # PREPARE SIM DICT FROM POTENTIALLY PRE-COMPUTED SIMILARITIES (if not already computed)
    sim_dict = {}
    if sim_type in PRECOMPUTED_SIMS:
      if sim_type == 'sbert':
        sim_dict = uq_from_similarity.prepare_sim_dict(sim_type)
      else:
        #dataset = 'codellama'
        dataset = 'spider_realistic_codellama' # -- hardcoded!
        sim_dict = uq_from_similarity.prep_and_save_sim_dict_ettubench(dataset, sim_type, save_bool=False)

        # # load the sim_dict
        # sim_dict_file_path = "sim_dict_data/"
        # sim_dict_file_name = 'sim_dict_codellama_' + sim_type + '.json' #---- hardcode filename
        #
        # INPUT_FILE = os.path.abspath(sim_dict_file_path + sim_dict_file_name)
        # with open(INPUT_FILE, 'r', encoding="utf8") as f:
        #   sim_dict = json.load(f)
        # print('Finished loading similarity dict!')
      print('Finished preparing similarity dict!')

    # Learn bayes params if needed
    if 'bayes' in uq_type:

      bayes_type = uq_type.split('-')[1]
      #bayes_setting = bayes_type.split(':')[0]
      sim_dist_type = bayes_type.split(':')[1]

      # remove samples -- WHY? CHECK THIS!
      #samples_data_valid = remove_samples_from_data(samples_data_valid, restrict_sample_indices_bool,
      #                                              accepted_sample_indices)

      acc_list_full, beta_data_given_correct, beta_data_given_incorrect, _ = \
        uq_from_similarity.prepare_bayes_data(samples_data_valid, sim_type, sim_dict, eps)

      # learn prior prob.
      p = np.mean(acc_list_full)

      # prepare bayes param dict based on uq_type (containing prior/post) and similarity distribution type
      bayes_param_dict = \
        uq_from_similarity.prepare_bayes_param_dict(split_index, uq_type, sim_dist_type, beta_data_given_correct,
                                                    beta_data_given_incorrect, eps)

      # for all cases including prior
      bayes_param_dict['p'] = p

      #print('bayes_param_dict:', bayes_param_dict)
      #print('mean_correct:', alpha_C / (alpha_C + beta_C))
      #print('mean_incorrect:', alpha_I / (alpha_I + beta_I))
      #print('\n')

    elif 'clf' in uq_type:

      clf_type = uq_type.split(':')[1]
      acc_list_full, X = \
        uq_from_similarity.prepare_clf_data(samples_data_valid, sim_type, sim_dict, eps)
      clf = uq_from_similarity.fit_classifier(acc_list_full, X, clf_type)


    # prepare dictionary with estimated confidence numbers
    if sim_type in valid_sim_types:
      conf_dict = \
        uq_from_similarity.prepare_conf_dict(samples_data_test, sim_type, sim_dict,
                                             uq_type, bayes_param_dict, clf, eps)
    else:
      raise ValueError(f'Invalid sim type {sim_type}')
  else:
    sim_type, uq_type = None, None
    conf_dict = None


  for sample_index in range(num_samples):

    accuracy_list = \
      [samples_data_test[query_index]['samples'][sample_index]['exec_acc'] for query_index in range(num_queries)]

    if setting == 'avg_prob':
      raw_score_list = [samples_data_test[query_index]['samples'][sample_index]['score']
                        for query_index in range(num_queries)]
      conf_list = [np.exp(score) for score in raw_score_list]
    elif setting == 'all_ones':
      conf_list = [1] * num_queries
    elif 'gens' in setting and sim_type in valid_sim_types:
      if conf_dict:
        conf_list = [conf_dict[query_index][sample_index] for query_index in range(num_queries)]
      else:
        raise ValueError(f'There should be a pre-computed confidence dictionary!')
    else:
      raise ValueError(f'Unknown setting specified {setting}')

    # evaluate metrics
    auroc, auarc, ece, ace, rmsce = evaluate_metrics(accuracy_list, conf_list, num_bins)

    exec_acc_vec.append(np.mean(accuracy_list))
    auroc_vec.append(auroc)
    auarc_vec.append(auarc)
    ece_vec.append(ece)
    ace_vec.append(ace)
    rmsce_vec.append(rmsce)

  # only consider acceptable sample_indices, if provided as input
  if restrict_sample_indices_bool and accepted_sample_indices is not None:

    #print('orig_num_samps:', len(exec_acc_vec))

    exec_acc_vec = [exec_acc_vec[i] for i in accepted_sample_indices]
    auroc_vec = [auroc_vec[i] for i in accepted_sample_indices]
    auarc_vec = [auarc_vec[i] for i in accepted_sample_indices]
    ece_vec = [ece_vec[i] for i in accepted_sample_indices]
    ace_vec = [ace_vec[i] for i in accepted_sample_indices]
    rmsce_vec = [rmsce_vec[i] for i in accepted_sample_indices]


    # exec_acc_vec = exec_acc_vec[0:num_accepted_sample_indices-1]
    # auroc_vec = auroc_vec[0:num_accepted_sample_indices-1]
    # auarc_vec = auarc_vec[0:num_accepted_sample_indices-1]
    # ece_vec = ece_vec[0:num_accepted_sample_indices-1]
    # ace_vec = ace_vec[0:num_accepted_sample_indices-1]

  #print('new_num_samps:', len(exec_acc_vec))

  mean_exec_acc = np.nanmean(exec_acc_vec)
  mean_auroc = np.nanmean(auroc_vec)
  mean_auarc = np.nanmean(auarc_vec)
  mean_ece = np.nanmean(ece_vec)
  mean_ace = np.nanmean(ace_vec)
  mean_rmsce = np.nanmean(rmsce_vec)

  return mean_exec_acc, mean_auroc, mean_auarc, mean_ece, mean_ace, mean_rmsce, \
         exec_acc_vec, auroc_vec, auarc_vec, ece_vec, ace_vec, rmsce_vec


# compute oracle AUARC from generated samples
def eval_samples_oracle_auarc(samples_data, restrict_sample_indices_bool = False, accepted_sample_indices = None):

  num_queries = len(samples_data)
  num_samples = len(samples_data[0]['samples'])

  oracle_auarc_vec = []

  for sample_index in range(num_samples):

    # only consider acceptable sample_indices, if provided as input
    if not restrict_sample_indices_bool or (restrict_sample_indices_bool and accepted_sample_indices is not None
                                            and sample_index in accepted_sample_indices):

      #print(sample_index)

      accuracy_list = \
        [samples_data[query_index]['samples'][sample_index]['exec_acc'] for query_index in range(num_queries)]

      #print(len(accuracy_list))
      #print(sum(accuracy_list))

      # use the accuracies as the confidences for the oracle (using ascending = False in the AUARC function)
      oracle_auarc  = compute_auarc(accuracy_list, accuracy_list)
      oracle_auarc_vec.append(oracle_auarc)

  #print(oracle_auarc_vec)
  # # only consider acceptable sample_indices, if provided as input
  # if restrict_sample_indices_bool and accepted_sample_indices is not None:
  #   oracle_auarc_vec = [oracle_auarc_vec[i] for i in accepted_sample_indices]
  #   #oracle_auarc_vec = oracle_auarc_vec[0:num_accepted_sample_indices-1]

  mean_oracle_auarc = np.nanmean(oracle_auarc_vec)

  return mean_oracle_auarc, oracle_auarc_vec


# perform eval across settings and while splitting the data potentially multiple times
def eval_multiple_test_sets_across_settings(result_file_name, num_test_samples, frac_test, samples_data, settings, num_bins = 10,
                                            restrict_sample_indices_bool = False, accepted_sample_indices = None):

  f = open(result_file_name + '.txt', 'w')

  for setting in settings:
    print('setting:', setting)
    eval_multiple_test_sets(f, num_test_samples, frac_test, samples_data, setting, num_bins,
                            restrict_sample_indices_bool, accepted_sample_indices)
    print('\n')

  f.close()

  return


# performs evaluation by taking multiple test sets
# num_test_samples is # of times the test set is sampled
# frac_test is the fraction of data that is considered to be test set
def eval_multiple_test_sets(f, num_test_samples, frac_test, samples_data, setting, num_bins = 10,
                            restrict_sample_indices_bool = False, accepted_sample_indices = None):

  f.write('Setting: ' + setting + '\r\n')
  f.write('----------------\r\n')

  num_queries = len(samples_data)
  num_instances_test_set = int(np.floor(frac_test * num_queries))
  num_instances_valid_set = num_queries - num_instances_test_set
  print('# of test instances:', num_instances_test_set)
  print('\n')

  mean_exec_vec, mean_auroc_vec, mean_auarc_vec, mean_ece_vec, mean_ace_vec, mean_rmsce_vec = [], [], [], [], [], []

  SEED = 10
  random.seed(SEED)

  for index in range(num_test_samples):
    # split into valid/test sets
    indices_queries = list(range(num_queries))
    random.shuffle(indices_queries)
    samples_data_valid = [samples_data[i] for i in indices_queries[:num_instances_valid_set]]
    samples_data_test = [samples_data[i] for i in indices_queries[num_instances_valid_set:]]

    # run evaluation on the valid/test splits
    split_index = index + 1
    mean_exec_acc, mean_auroc, mean_auarc, mean_ece, mean_ace, mean_rmsce, _, _, _, _, _, _ = \
      eval_samples(split_index, samples_data_valid, samples_data_test, setting, num_bins,
                   restrict_sample_indices_bool, accepted_sample_indices)
    mean_exec_vec.append(mean_exec_acc)
    mean_auroc_vec.append(mean_auroc)
    mean_auarc_vec.append(mean_auarc)
    mean_ece_vec.append(mean_ece)
    mean_ace_vec.append(mean_ace)
    mean_rmsce_vec.append(mean_rmsce)


  avg_exec_acc = np.mean(mean_exec_vec)
  exec_acc_error = (max(mean_exec_vec)-min(mean_exec_vec))/2
  avg_auroc = np.mean(mean_auroc_vec)
  auroc_error = (max(mean_auroc_vec) - min(mean_auroc_vec))/2
  avg_auarc = np.mean(mean_auarc_vec)
  auarc_error = (max(mean_auarc_vec) - min(mean_auarc_vec))/2
  avg_ece = np.mean(mean_ece_vec)
  ece_error = (max(mean_ece_vec) - min(mean_ece_vec))/2
  avg_ace = np.mean(mean_ace_vec)
  ace_error = (max(mean_ace_vec) - min(mean_ace_vec))/2
  avg_rmsce = np.mean(mean_rmsce_vec)
  rmsce_error = (max(mean_rmsce_vec) - min(mean_rmsce_vec)) / 2

  print('Exec_acc: ', avg_exec_acc, '+/-', exec_acc_error)
  print('AUROC: ', avg_auroc, '+/-', auroc_error)
  print('AUARC: ', avg_auarc, '+/-', auarc_error)
  print('ECE: ', avg_ece, '+/-', ece_error)
  print('ACE: ', avg_ace, '+/-', ace_error)
  print('RMSCE: ', avg_rmsce, '+/-', rmsce_error)

  f.write('Exec_acc: ' + str(avg_exec_acc) + '+/-' + str(exec_acc_error) + '\r\n')
  f.write('AUROC: ' + str(avg_auroc) + '+/-' + str(auroc_error) + '\r\n')
  f.write('AUARC: ' + str(avg_auarc) + '+/-' + str(auarc_error) + '\r\n')
  f.write('ECE: ' + str(avg_ece) + '+/-' + str(ece_error) + '\r\n')
  f.write('ACE: ' + str(avg_ace) + '+/-' + str(ace_error) + '\r\n')
  f.write('RMSCE: ' + str(avg_rmsce) + '+/-' + str(rmsce_error) + '\r\n')

  f.write('\r\n----------------\r\n')




  return


# run eval for multiple settings, provided as a list
def eval_samples_multiple_approaches(samples_data_valid, samples_data_test, settings, num_bins = 10,
                                     restrict_sample_indices_bool = False, accepted_sample_indices = None):

  mean_oracle_auarc, oracle_auarc_vec = \
    eval_samples_oracle_auarc(samples_data_test, restrict_sample_indices_bool, accepted_sample_indices)
  print('\n')
  print('Avg. oracle AUARC:', mean_oracle_auarc)


  for setting in settings:
    print('\n')
    print('Setting:', setting)
    print('----------------')
    mean_exec_acc, mean_auroc, mean_auarc, mean_ece, mean_ace, mean_rmsce, exec_acc_vec, auroc_vec, auarc_vec,\
    ece_vec, ace_vec, rmsce_vec = \
      eval_samples(samples_data_valid, samples_data_test, setting, num_bins,
                   restrict_sample_indices_bool, accepted_sample_indices)
    print('mean_exec_acc:', mean_exec_acc)
    print('mean_auroc:', mean_auroc)
    print('mean_auarc:', mean_auarc)
    print('mean_ece:', mean_ece)
    print('mean_ace:', mean_ace)
    print('mean_rmsce:', mean_rmsce)

  return


# function computes area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
def compute_auarc(a, u):

  # num_queries = len(accuracy_list)
  # tot_correct = sum(accuracy_list) # total number of correct queries
  #
  # # sort accuracies in increasing order of confidences -- we reject one at a time starting from the least confident
  # acc_sorted_by_conf = [x for _, x in sorted(zip(conf_list, accuracy_list))]
  #
  # acc_points_auarc_curve = [tot_correct/num_queries] # initial point is acc. w/o rejection
  # frac_reject_points_auarc_curve = [0] # initially no rejection
  #
  # remaining_correct = tot_correct # initialization of remaining points that are correct
  # for index in range(num_queries):
  #   num_accepted_points = num_queries - (index+1)
  #   # print(num_accepted_points)
  #   remaining_correct -= acc_sorted_by_conf[index] # update remaining number of correct queries
  #   # print(remaining_correct)
  #   this_acc = remaining_correct/num_accepted_points if num_accepted_points > 0 else 1  # update accuracy
  #   this_frac_reject = 1 - (num_accepted_points/num_queries) # update the fraction of rejected points
  #
  #   acc_points_auarc_curve.append(this_acc)
  #   frac_reject_points_auarc_curve.append(this_frac_reject)
  #
  # auarc = metrics.auc(frac_reject_points_auarc_curve, acc_points_auarc_curve)

  # return auarc, frac_reject_points_auarc_curve, acc_points_auarc_curve

  # create pandas dataframe and do a random shuffle of rows due to the issue with AUARC around identical confidences
  df = pd.DataFrame({"u": u, 'a': a}).sample(frac=1)
  # sort by confidences and compute the metric
  df = df.sort_values('u', ascending=False)  # check if ascending should be true or false!!!!
  df['amean'] = df['a'].expanding().mean()
  auarc = metrics.auc(np.linspace(0, 1, len(df)), df['amean'])

  return auarc


################################
# EXPERIMENTS
################################


def run_temp_first_exps(dataset, sampling_type, split, temperature, num_test_samples, frac_test, num_bins,
                        restrict_sample_indices_bool, accepted_sample_indices):

  #settings -- 5 sim metrics and 6-8 aggregation approaches
  settings = ['gens|output_type|agg-arith', 'gens|jaccard|agg-arith', 'gens|rouge1|agg-arith',
              'gens|rougeL|agg-arith', # 'gens|sbert|agg-arith',
              #'gens|output_type|agg-geom', 'gens|jaccard|agg-geom', 'gens|rouge1|agg-geom',
              #'gens|rougeL|agg-geom', # 'gens|sbert|agg-geom',
              #'gens|output_type|agg-harm', 'gens|jaccard|agg-harm', 'gens|rouge1|agg-harm',
              #'gens|rougeL|agg-harm', # 'gens|sbert|agg-harm',
              'gens|output_type|bayes-post:beta', 'gens|jaccard|bayes-post:beta', 'gens|rouge1|bayes-post:beta',
              'gens|rougeL|bayes-post:beta', # 'gens|sbert|bayes-post:beta',
              #'gens|output_type|bayes-post:mixed', 'gens|jaccard|bayes-post:mixed', 'gens|rouge1|bayes-post:mixed',
              #'gens|rougeL|bayes-post:mixed', 'gens|sbert|bayes-post:mixed',
              'gens|output_type|spec-ecc', 'gens|jaccard|spec-ecc', 'gens|rouge1|spec-ecc',
              'gens|rougeL|spec-ecc', # 'gens|sbert|spec-ecc',
              'gens|output_type|clf:lr', 'gens|jaccard|clf:lr', 'gens|rouge1|clf:lr',
              'gens|rougeL|clf:lr', #'gens|sbert|clf:lr',
              'gens|output_type|clf:rf', 'gens|jaccard|clf:rf', 'gens|rouge1|clf:rf',
              'gens|rougeL|clf:rf' #, 'gens|sbert|clf:rf'
              ]

  # same as above but fewer settings
  # settings = ['gens|sbert|spec-ecc', 'gens|sbert|agg-arith', 'gens|sbert|bayes-post:beta',
  #             'gens|sbert|clf:lr', 'gens|sbert|clf:rf']

  # settings = ['gens|aligon|spec-ecc', 'gens|aligon|agg-arith', 'gens|aligon|bayes-post:beta',
  #             'gens|aligon|clf:lr', 'gens|aligon|clf:rf',
  #             'gens|aouiche|spec-ecc', 'gens|aouiche|agg-arith', 'gens|aouiche|bayes-post:beta',
  #             'gens|aouiche|clf:lr', 'gens|aouiche|clf:rf',
  #             'gens|makiyama|spec-ecc', 'gens|makiyama|agg-arith', 'gens|makiyama|bayes-post:beta',
  #             'gens|makiyama|clf:lr', 'gens|makiyama|clf:rf'
  #             ]


  #settings = ['gens|aligon|agg-arith']



  # settings = ['all_ones', 'avg_prob', 'gens|jaccard|bayes-prior:beta',
  #             'gens|jaccard|agg-arith', 'gens|jaccard|clf:lr', 'gens|jaccard|clf:rf',
  #             'gens|rougeL|agg-arith', 'gens|rougeL|clf:lr', 'gens|rougeL|clf:rf'
  # ]


  # # small number of cases for temp-all
  # settings = ['all_ones', 'avg_prob',
  #             'gens|jaccard|bayes-prior:beta',
  #             'gens|jaccard|agg-arith', 'gens|jaccard|clf:rf',  #'gens|jaccard|clf:lr',
  #             'gens|rougeL|agg-arith', 'gens|rougeL|clf:rf'
  #             #, 'gens|rougeL|clf:lr'
  #             ]

  # settings = [
  #   'gens|aouiche|agg-arith', 'gens|makiyama|agg-arith',
  #   'gens|aligon|clf:rf', 'gens|aouiche|clf:rf', 'gens|makiyama|clf:rf'
  # ]

  # settings = ['gens|jaccard|agg-arith']

  #settings = ['gens|output_type|agg-arith', 'gens|rougeL|agg-arith']


  # load data
  if dataset == 'MDE55':
    # load MDE data
    samples_data = load_samples_data_MDE55(temperature)
  elif dataset == 'spider_deepseeker':
    # load and process Spider data with deepseeker output
    samples_data = load_and_process_samples_data_spider_deepseeker(sampling_type)
  elif dataset == 'spider_fp' or dataset == 'spider_fine-tuned_granite':
    samples_data = load_and_process_samples_data_spider_csv(dataset)
  elif dataset == 'spider_codellama' or dataset == 'spider_granite' or dataset == 'spider_realistic_codellama':
    # load and process Spider data with fewshot output 
    samples_data = \
      load_and_process_samples_data_spider_codellama_fewshot(dataset, sampling_type, split, temperature)
  elif dataset == 'bird_granite':
    # load and process Bird data with fine-tuned model output
    samples_data = load_and_process_samples_data_bird()
  else:
    raise ValueError('This dataset is not covered!')

  print('No. of queries:', len(samples_data))

  result_file_name = dataset + '_' + sampling_type

  eval_multiple_test_sets_across_settings(result_file_name, num_test_samples, frac_test, samples_data,
                                          settings, num_bins, restrict_sample_indices_bool, accepted_sample_indices)

  return



def train_conf_est_model_and_prep_conf_dict(samples_data_valid, samples_data_test, setting, eps, valid_sim_types):

  bayes_param_dict = {}
  clf = None

  # for similarity-based approaches, compute a dictionary with confidences
  if 'gens' in setting:
    sim_type, uq_type = setting.split('|')[1], setting.split('|')[2]

    # PREPARE SIM DICT FROM POTENTIALLY PRE-COMPUTED SIMILARITIES (if not already computed)
    sim_dict = {}
    if sim_type in PRECOMPUTED_SIMS:
      if sim_type == 'sbert':
        sim_dict = uq_from_similarity.prepare_sim_dict(sim_type)
      else:
        dataset = 'codellama'
        sim_dict = uq_from_similarity.prep_and_save_sim_dict_ettubench(dataset, sim_type, save_bool=False)

        # # load the sim_dict
        # sim_dict_file_path = "sim_dict_data/"
        # sim_dict_file_name = 'sim_dict_codellama_' + sim_type + '.json' #---- hardcode filename
        #
        # INPUT_FILE = os.path.abspath(sim_dict_file_path + sim_dict_file_name)
        # with open(INPUT_FILE, 'r', encoding="utf8") as f:
        #   sim_dict = json.load(f)
        # print('Finished loading similarity dict!')
      print('Finished preparing similarity dict!')

    # Learn bayes params if needed
    if 'bayes' in uq_type:
      bayes_type = uq_type.split('-')[1]
      # bayes_setting = bayes_type.split(':')[0]
      sim_dist_type = bayes_type.split(':')[1]

      acc_list_full, beta_data_given_correct, beta_data_given_incorrect, _ = \
        uq_from_similarity.prepare_bayes_data(samples_data_valid, sim_type, sim_dict, eps)

      # learn prior prob.
      p = np.mean(acc_list_full)

      # prepare bayes param dict based on uq_type (containing prior/post) and similarity distribution type
      split_index = None
      bayes_param_dict = \
        uq_from_similarity.prepare_bayes_param_dict(split_index, uq_type, sim_dist_type, beta_data_given_correct,
                                                    beta_data_given_incorrect, eps)

      # for all cases including prior
      bayes_param_dict['p'] = p

    elif 'clf' in uq_type:

      clf_type = uq_type.split(':')[1]
      acc_list_full, X = \
        uq_from_similarity.prepare_clf_data(samples_data_valid, sim_type, sim_dict, eps)
      clf = uq_from_similarity.fit_classifier(acc_list_full, X, clf_type)

    # prepare dictionary with estimated confidence numbers
    if sim_type in valid_sim_types:
      conf_dict = \
        uq_from_similarity.prepare_conf_dict(samples_data_test, sim_type, sim_dict,
                                             uq_type, bayes_param_dict, clf, eps)
    else:
      raise ValueError(f'Invalid sim type {sim_type}')
  else:
    #sim_type, uq_type = None, None
    conf_dict = None

  return conf_dict


# provide confidence estimate information to queries after splitting
def run_conf_estimates_exp(samples_data, frac_test, num_test_samples):

  #setting = 'gens|jaccard|agg-arith'

  eps = 0.0001  # hardcoded for now

  # valid list of sim_types, i.e. those that are enabled
  valid_sim_types = ['output_type', 'jaccard', 'rouge1', 'rouge2', 'rougeL', 'sbert', 'aligon', 'aouiche', 'makiyama']

  num_queries = len(samples_data)
  num_instances_test_set = int(np.floor(frac_test * num_queries))
  num_instances_valid_set = num_queries - num_instances_test_set
  print('# of test instances:', num_instances_test_set)
  print('\n')

  output_dict = {}

  SEED = 10
  random.seed(SEED)

  for split_index in range(1, num_test_samples+1):

    # split into valid/test sets
    indices_queries = list(range(num_queries))
    random.shuffle(indices_queries)
    samples_data_valid = [samples_data[i] for i in indices_queries[:num_instances_valid_set]]
    samples_data_test = [samples_data[i] for i in indices_queries[num_instances_valid_set:]]

    # obtain confidence estimates for all methods considered -- manually entering the settings
    setting = 'gens|jaccard|agg-arith'
    conf_dict_1 = \
      train_conf_est_model_and_prep_conf_dict(samples_data_valid, samples_data_test, setting, eps, valid_sim_types)
    setting = 'gens|rougeL|agg-arith'
    conf_dict_2 = \
      train_conf_est_model_and_prep_conf_dict(samples_data_valid, samples_data_test, setting, eps, valid_sim_types)
    # setting = 'gens|jaccard|clf:rf'
    # conf_dict_3 = \
    #   train_conf_est_model_and_prep_conf_dict(samples_data_valid, samples_data_test, setting, eps, valid_sim_types)
    # setting = 'gens|rougeL|clf:rf'
    # conf_dict_4 = \
    #   train_conf_est_model_and_prep_conf_dict(samples_data_valid, samples_data_test, setting, eps, valid_sim_types)

    print('Finished preparing all confidences dicts for split index:', split_index)

    # if conf_dict is None or len(conf_dict) == 0:
    #   raise ValueError('Something is wrong with conf. dict preparation!')

    this_split_dict = samples_data_test
    for q_index in range(len(samples_data_test)):

      this_q_info = samples_data_test[q_index]
      this_q_samples = this_q_info['samples']

      setting = 'gens|jaccard|agg-arith'
      conf_list_this_q_1 = conf_dict_1[q_index]
      # append the score information from confidence estimation
      for sample_index in range(len(this_q_samples)):
        this_q_samples[sample_index]['score_' + setting] = conf_list_this_q_1[sample_index]
      setting = 'gens|rougeL|agg-arith'
      conf_list_this_q_2 = conf_dict_2[q_index]
      # append the score information from confidence estimation
      for sample_index in range(len(this_q_samples)):
        this_q_samples[sample_index]['score_' + setting] = conf_list_this_q_2[sample_index]
      # setting = 'gens|jaccard|clf:rf'
      # conf_list_this_q_3 = conf_dict_3[q_index]
      # # append the score information from confidence estimation
      # for sample_index in range(len(this_q_samples)):
      #   this_q_samples[sample_index]['score_' + setting] = conf_list_this_q_3[sample_index]
      # setting = 'gens|rougeL|clf:rf'
      # conf_list_this_q_4 = conf_dict_4[q_index]
      # # append the score information from confidence estimation
      # for sample_index in range(len(this_q_samples)):
      #   this_q_samples[sample_index]['score_' + setting] = conf_list_this_q_4[sample_index]

      #conf_list_this_q = conf_dict[q_index]
      #print('conf_list:', conf_list_this_q)
      # # append the score information from confidence estimation
      # for sample_index in range(len(this_q_samples)):
      #   this_q_samples[sample_index]['score_' + setting] = conf_list_this_q[sample_index]

      this_split_dict[q_index]['samples'] = this_q_samples

    output_dict[split_index] = this_split_dict

  # save output file


  return output_dict



















