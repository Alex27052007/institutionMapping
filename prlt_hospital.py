import time
import math
import recordlinkage
from recordlinkage.preprocessing import clean
import pandas
from recordlinkage.base import BaseCompareFeature
import re
import numpy as np

parts_to_remove = '\.|\.|gGmbH|GmbH'


def clean_name(row):
    name_string = row['name']
    if isinstance(name_string,str):
        cleaned_name = re.sub(parts_to_remove, '', name_string)
        return cleaned_name
    else:
        return ''
def clean_Name(row):
    name_string = row['Name']
    if isinstance(name_string,str):
        cleaned_name = re.sub(parts_to_remove, '', name_string)
        return cleaned_name
    else:
        return ''

def save_results(links_pred, customer_hospitals, alberta_hospitals):
    pd_result = pandas.merge(customer_hospitals, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_hospitals, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'Adresse': 'Adresse Kunde', 'PLZ': 'PLZ Kunde', 'Ort': 'Ort Kunde', 
    'name': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    #del pd_result["institution_name"]
    del pd_result["institution_address"]
    del pd_result["Name_clean"]

    pd_result.to_csv('results_matches_hospital.csv')
  
   
def get_block_indexer():
   indexer = recordlinkage.Index()
   indexer.block(left_on='PLZ', right_on='postalCode')
   return indexer

def get_comparer(methodName):
    comparer = recordlinkage.Compare()
    #comparer.string('Name_clean', 'name_clean', method=methodName, label='institution_name')
    comparer.string('Adresse', 'address', method=methodName, label='institution_address')
    comparer.exact('PLZ', 'postalCode', label='institution_postalcode')

    return comparer

start_time = time.time()
alberta_hospitals = pandas.read_csv('alberta/Alle_Kliniken_Alberta.csv',  sep=';', index_col='_id', dtype={ 'postalCode': str,})
customer_hospitals = pandas.read_csv('customer/hospital_mwm.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})

#Cleaning
customer_hospitals['Name_clean'] = customer_hospitals.apply(clean_Name, axis=1)
alberta_hospitals['name_clean'] = alberta_hospitals.apply(clean_name, axis=1)

start_time = time.time()
# Indexation step
indexer = get_block_indexer()

candidate_links = indexer.index(customer_hospitals, alberta_hospitals)
print('Potential candidates', len(candidate_links))
# Comparison step
comparer = get_comparer('qgram')
comparison_vectors = comparer.compute(candidate_links, customer_hospitals, alberta_hospitals)

# Classification step
scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[60, 40])
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.82]
end_time = time.time()
duration = end_time - start_time

matches.to_csv('predictions_ho.csv')
links_pred = pandas.read_csv('predictions_ho.csv', sep=',', index_col='Nr')

count_matches = len(links_pred)
count_nonmatches = len(customer_hospitals) - len(links_pred)

save_results(links_pred.copy(), customer_hospitals, alberta_hospitals)

# Evaluation
print('Count matches', count_matches)
print('Count non-matches', count_nonmatches)
print('Duration', end_time - start_time)


