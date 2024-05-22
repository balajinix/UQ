"""
  File:             sampling.py

  About
  ------------------
  functions for sampling generations for UQ

"""

import requests, json
import sys, os
import pandas as pd
import io
from collections import OrderedDict
from enum import Enum
import pprint
pp = pprint.PrettyPrinter(indent=4)
import traceback
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from typing import List, Dict, Tuple, Set, Union


SCHEMA_NAME = "CSTINSIGHT"
MDE_SCHEMA_PATH = "../../data/MDE/mde_table_schema"
GT_INFO_PATH = "../../data/MDE/"
GT_INFO_FILE = "cryptic_column_deepseek-coder-6.7b-tuned_5678_verified.json"

DB2_EXECUTOR_REST_API_HOST = os.environ['HOST_NAME']
DB2_EXECUTOR_REST_API_PORT = 7020 # also 7024


class ExecutionStatus(Enum):
    correct_superset = "CorrectSuperSet"
    correct_exact = "CorrectExact"
    no_error = "NoError_NotMatch"
    syntax_error = "SyntaxError"
    execution_error = "ExecutionError"


# get all tables related to the MDE schema
def get_all_tables_MDE():
  tables = os.listdir(MDE_SCHEMA_PATH)
  tables = [x for x in tables if x.startswith(SCHEMA_NAME)]
  # append all tables schema
  ori_schema = []
  for file in os.listdir(MDE_SCHEMA_PATH):
    ori_schema.append(json.load(open(os.path.join(MDE_SCHEMA_PATH, file))))

  return tables, ori_schema


# get MDE NL questions, ground truth SQL and relevant tables
def get_questions_and_gt_info_MDE():

  # read file
  INPUT_FILE = os.path.abspath(GT_INFO_PATH + GT_INFO_FILE)
  with open(INPUT_FILE, 'r', encoding="utf8") as f:
    data = json.load(f)

  question_and_gt_info = []
  for q in data:
    this_dict = {
      'question': q['question'],
      'groundtruth_sql': q['groundtruth_sql'],
      'related_tables': q['related_tables']
    }
    question_and_gt_info.append(this_dict)

  return question_and_gt_info


# run schema linker
def schema_linker_request(db_schema, nl_question, qualified_tables=[]):

  my_json = {"DatabaseSchema": {'tables': db_schema}, "NLQuestion": nl_question, "QualifiedTables": qualified_tables}
  HOST = os.environ['HOST_NAME']
  url = f"https://{HOST}:7018/generate"
  headers = {"Content-Type": "application/json", "accept": "application/json"}
  r = requests.post(url, headers=headers, json=my_json, verify=False)
  if r.status_code == 200:
    return r.json()
  else:
    raise ValueError(f"Cannot request schema linker, got {r.status_code} code")


# filter schema links
def filter_schema_links(schema_links: List[Tuple[str, float]], threshold: float = 0.5,
                        schema_top_k_min: int = 3, schema_top_k_max: int = 30) -> List[str]:

  schema_links.sort(key=lambda x: x[1], reverse=True)
  # links above threshold or at least top_k_min, but at most top_k_max
  schema_links_filtered = [qc for qc, score in schema_links if score >= threshold]
  score_filtered = [score for qc, score in schema_links if score >= threshold]
  if len(schema_links_filtered) < schema_top_k_min:
    schema_links_filtered = [qc for qc, score in schema_links][:schema_top_k_min]
    score_filtered = [score for qc, score in schema_links][:schema_top_k_min]
  elif len(schema_links_filtered) > schema_top_k_max:
    schema_links_filtered = schema_links_filtered[:schema_top_k_max]
    score_filtered = score_filtered[:schema_top_k_max]
  return schema_links_filtered, score_filtered


# Apply SQL gen API
def db2_exec_request(query, schema_name="CSTINSIGHT"):

  my_json = {"SQLQuery": query, "Schema": schema_name}
  HOST = os.environ['HOST_NAME']
  url = f"https://{HOST}:7020/generate"
  headers = {"Content-Type": "application/json", "accept": "application/json"}
  r = requests.post(url, headers=headers, json=my_json, verify=False)
  if r.status_code == 200:
      return r.json()
  else:
      raise ValueError(f"Cannot execute query, got {r.status_code} code")


# get sample of table first 3 rows
def get_table_column_values(tables):
  # get sample data
  my_dict = {}
  for table_name in tables:
     # table_name = "CSTINSIGHT.CODE"
    query_cmd = f"SELECT * FROM {table_name};"
    res = db2_exec_request(query_cmd)

    #print(table_name)

    if "." in table_name:
       table_name = table_name.split(".")[-1]
    # get dataframe
    df = pd.read_json(io.StringIO(res["results"])).head(3)

    # need to replace NaN values before json serialization
    df = df.fillna('')

    # print(df)
    my_dict[table_name.upper()] = {}
    for column in df.columns:
       my_dict[table_name.upper()][column.upper()] = df[column].to_list()

  return my_dict


def sql_gen_request(db_schema, nl_question, table_col_vals, qualified_columns=[], generate_config=None):
  my_json = {"DatabaseSchema": {"tables": db_schema},
             "NLQuestion": nl_question,
             "QualifiedColumns": qualified_columns,
             "TableColumnValues": table_col_vals,
             "GenerateConfig": generate_config}
  HOST = os.environ['HOST_NAME']
  url = f"https://{HOST}:7019/generate"
  headers = {"Content-Type": "application/json", "accept": "application/json"}
  r = requests.post(url, headers=headers, json=my_json, verify=False)
  if r.status_code == 200:
    return r.json()
  else:
    raise ValueError(f"Cannot request SQL generator, got {r.status_code} code")


def verify_sql(g_str: str,
               sql_texts: List[str],
               db2_executor_rest_api_host: str = None,
               db2_executor_rest_api_port: int = None,
               ) -> Tuple[List, List]:

  g_status, g_res_df = execute_raw_sql(g_str, db2_executor_rest_api_host, db2_executor_rest_api_port)
  execution_details = [g_res_df.head().to_json(orient='records', date_format='iso')]

  if g_status != ExecutionStatus.no_error:
    print(f'Groundtruth execution exception: {g_res_df}')
    return None, execution_details

  execution_results = []
  # remove duplicated columns for groundtruth df
  g_res_df = remove_duplicate_col_df(g_res_df)
  for p_str_i in sql_texts:
    execution_result, p_res_df = execute_raw_sql(p_str_i, db2_executor_rest_api_host, db2_executor_rest_api_port)
    if execution_result == ExecutionStatus.no_error:
      if g_res_df is not None and p_res_df is not None:
        # remove duplicated columns for model generated df
        p_res_df = remove_duplicate_col_df(p_res_df)
        if compare_df(g_res_df, p_res_df) == 0:
          execution_result = ExecutionStatus.correct_exact
        elif compare_df(g_res_df, p_res_df) == -1:
          execution_result = ExecutionStatus.correct_superset

    try:
      execution_results.append(execution_result)
      #execution_details.append(p_res_df.head().to_json(orient='records', date_format='iso'))
    except Exception:
      print(traceback.format_exc())
      pass

  return execution_results, execution_details


def execute_raw_sql(
        sql_str: str,
        db2_executor_rest_api_host: str = None,
        db2_executor_rest_api_port: int = None,
) -> Tuple[ExecutionStatus, Union[str, pd.DataFrame]]:
  returned_code, returned_msg = db2_run_sql(
    sql_query=sql_str,
    db2_executor_rest_api_host=db2_executor_rest_api_host,
    db2_executor_rest_api_port=db2_executor_rest_api_port
  )

  if returned_code:
    return ExecutionStatus.execution_error, returned_msg

  return ExecutionStatus.no_error, pd.DataFrame(json.loads(returned_msg['results']))


def db2_run_sql(
        sql_query,
        schema_name="CSTINSIGHT",
        retries=30,
        db2_executor_rest_api_host: str = 'localhost',
        db2_executor_rest_api_port: int = 7020
) -> Tuple[int, str]:
  assert db2_executor_rest_api_host is not None
  assert db2_executor_rest_api_port is not None

  my_json = {"SQLQuery": sql_query, "Schema": schema_name}
  # url = f"https://{HOST}:7020/generate"
  url = f"https://{db2_executor_rest_api_host}:{db2_executor_rest_api_port}/generate"
  headers = {"Content-Type": "application/json", "accept": "application/json"}
  for i in range(retries):
    r = requests.post(url, headers=headers, json=my_json, verify=False)
    if r.status_code == 200:
      if 'Error' in r.json()['msg']:
        # return 1, r.json()['results'] # 'results' is the error message
        return 1, r.json()

      # jstr = json.loads(r.json()['results'])
      return 0, r.json()

    time.sleep(i)
    # raise ValueError(f"Cannot request DB2 after {retries} retries, got {r.status_code} code")
  return 1, {"results": f"Cannot request DB2 after {retries} retries, got {r.status_code} code"}


# for verification
def remove_duplicate_col_df(df):
    return df.loc[:,~df.columns.duplicated()]

def convert_df_to_set(df) -> Set:
  # remove duplicate columns
  df = remove_duplicate_col_df(df)
  # convert dataframe to set of tuples (cannot be set of list due to unhashable error)
  return set([tuple(sorted(df[c].to_list(), key=lambda x: (x is None, x))) for c in df.columns.values])

def compare_df(df1, df2) -> bool:
    set1 = convert_df_to_set(df1)
    set2 = convert_df_to_set(df2)

    # -1: df1 is subset of df2
    # 0: df1 == df2
    # 1: df2 is subset of df1
    return -1 if (set1 < set2) else 0 if (set1 == set2) else 1



def gen_samples_single_query_MDE(nl_question, gt_sql, qualified_tables, db_schema, num_samples, temperature, threshold):

  this_query_dict = {
    'question': nl_question,
    'gt_sql': gt_sql
  }

  ## get tables and original schema
  #tables, ori_schema = get_all_tables_MDE()
  ## identify NL questions as well as the ground truth SQL and relevant tables (assume gold)
  #question_and_gt_info = get_questions_and_gt_info_MDE()

  # run schema linker
  response = schema_linker_request(db_schema, nl_question)
  #pp.pprint(response)

  # filter schema linker based on pre-determined threshold
  schema_links_filtered, score_filtered = filter_schema_links(list(response["results"].items()), threshold)
  col_score_tuples = [(x, y) for x, y in zip(schema_links_filtered, score_filtered)]

  # get 3-row sample for each table
  sub_table_col_vals = get_table_column_values(qualified_tables)
  #pp.pprint(sub_table_col_vals)

  qualified_columns = [x for (x, y) in col_score_tuples]
  #pp.pprint(qualified_columns)

  # run text2sql for multiple samples
  generate_config = {
    "num_return_sequences": num_samples,
    "max_new_tokens": 150,
    "temperature": temperature
    # "top_k": 50,
    # "top_p": 0.9,
    # "no_repeat_ngram_size": 0,
  }

  response = sql_gen_request(db_schema, nl_question, sub_table_col_vals, qualified_columns=qualified_columns,
                             generate_config=generate_config)
  # pp.pprint(response)
  #pp.pprint(response["results"])

  this_query_list = []
  for gen in response["results"]:
    gen_sql = gen['text']
    gen_score = gen['score']
    execution_results, _ = verify_sql(gt_sql, [gen_sql], DB2_EXECUTOR_REST_API_HOST, DB2_EXECUTOR_REST_API_PORT)
    if execution_results[0].value == 'CorrectExact':
      exec_acc = 1
    else:
      exec_acc = 0
    this_gen_dict = {
      'gen_sql': gen_sql,
      'score': gen_score,
      'exec_acc': exec_acc
    }
    this_query_list.append(this_gen_dict)

  this_query_dict['samples'] = this_query_list

  return this_query_dict


def gen_samples_MDE(num_samples, temperature, threshold):

  output = []

  # get tables and original schema
  tables, ori_schema = get_all_tables_MDE()

  # identify NL questions as well as the ground truth SQL and relevant tables (assume gold)
  question_and_gt_info = get_questions_and_gt_info_MDE()

  # loop over all queries
  for index in range(len(question_and_gt_info)):

    nl_question = question_and_gt_info[index]['question']
    gt_sql = question_and_gt_info[index]['groundtruth_sql']
    qualified_tables = question_and_gt_info[index]['related_tables']

    # restrict schema to relevant tables
    db_schema = [x for x in ori_schema if x['name'] in qualified_tables]
    # generate samples for this query and execute all generated queries
    this_query_dict = gen_samples_single_query_MDE(nl_question, gt_sql, qualified_tables,
                                                   db_schema, num_samples, temperature, threshold)
    print('Sampling complete for query:', index+1)
    output.append(this_query_dict)

  return output



################################
# SAMPLING WITH IN-HOUSE MODEL
################################








################################
# LOAD AND SAVE
################################





