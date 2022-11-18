import time
import math
import recordlinkage
from recordlinkage.index import Full
from recordlinkage.preprocessing import clean
import pandas
from recordlinkage.base import BaseCompareFeature
import re
import numpy as np

parts_to_remove = '\.|\.|Die Gesundheitskasse|Private Krankenversicherung|Krankenversicherung|Die Gesundheitskasse für Sachsen und Thüringen'




def save_results(links_pred, customer_payers, alberta_payers):
    pd_result = pandas.merge(customer_payers, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_payers, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'name': 'Name Alberta'})

    pd_result.to_csv('results_matches.csv')
  
  
def get_full_indexer():
    indexer = recordlinkage.Index()
    indexer.add(Full())
    return indexer

def get_block_indexer():
   indexer = recordlinkage.Index()
   indexer.block(left_on='name', right_on='Name')
   return indexer

def get_comparer(methodName):
    comparer = recordlinkage.Compare()
    comparer.string('Name', 'name', method=methodName, label='institution_name')

    return comparer

start_time = time.time()
alberta_payers = pandas.read_csv('alberta/Alle_KK_Alberta.csv',  sep=';', index_col='_id')
customer_payers = pandas.read_csv('customer/payer_medhuman.csv',  sep=';', index_col='Nr')

#Cleaning
start_time = time.time()
# Indexation step
indexer =  get_full_indexer()

candidate_links = indexer.index(customer_payers, alberta_payers)
print('Potential candidates', len(candidate_links))
# Comparison step
comparer = get_comparer('qgram')
comparison_vectors = comparer.compute(candidate_links, customer_payers, alberta_payers)

# Classification step
scores = np.average(
    comparison_vectors.values,
    axis=1)
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.82]
end_time = time.time()
duration = end_time - start_time

matches.to_csv('predictions_kk.csv')
links_pred = pandas.read_csv('predictions_kk.csv', sep=',', index_col='Nr')

count_matches = len(links_pred)
count_nonmatches = len(customer_payers) - len(links_pred)

save_results(links_pred.copy(), customer_payers, alberta_payers)

# Evaluation
print('Count matches', count_matches)
print('Count non-matches', count_nonmatches)
print('Duration', end_time - start_time)


