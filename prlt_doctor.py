import time
import math
import recordlinkage
from recordlinkage.preprocessing import clean
import pandas as pd
from recordlinkage.base import BaseCompareFeature
import re
import numpy as np

parts_to_remove = '\.|\.|Herr|Frau|Dr|Drs|med|medic|univ|Dipl|Prof|Priv|Doz|Praxis|Gemeinschaftspraxis|GP|rer|soc|Facharzt|Klinik|Kardiologie|Urologie|Schlaflabor|Hausarzt'

def clean_Name(row):
    name_string = row['Name']
    if isinstance(name_string,str):
        cleaned_name = re.sub(parts_to_remove, '', name_string)
        return cleaned_name
    else:
        return ''

def save_results(links_pred, customer_doctors, alberta_doctors):
    pd_result = pd.merge(customer_doctors, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_doctors, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'Adresse': 'Adresse Kunde', 'PLZ': 'PLZ Kunde', 'Ort': 'Ort Kunde', 
    'name': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    del pd_result["institution_address"]
    del pd_result["Name_clean"]

    pd_result.to_csv('results_matches_doctor.csv')
  
   
def get_block_indexer():
    indexer = recordlinkage.Index()
    indexer.block(left_on='PLZ', right_on='postalCode')
    indexer.block(left_on='Name', right_on='fullName')
    return indexer

def get_comparer(methodName):
    comparer = recordlinkage.Compare()
    comparer.string('Name_clean', 'fullName', method=methodName, label='institution_name')
    comparer.string('PLZ', 'postalCode', method=methodName, label='institution_postalCode')
    comparer.string('Adresse', 'address', method=methodName, label='institution_address')

    return comparer

start_time = time.time()
alberta_doctors = pd.read_csv('./alberta/Alle_Ärzte_Alberta.csv',  sep=';', index_col='_id', dtype={'gender': str, 'postalCode': str,})
customer_doctors = pd.read_csv('./customer/doctors_medhuman.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})

#Cleaning
customer_doctors['Name_clean'] = customer_doctors.apply(clean_Name, axis=1)

start_time = time.time()
# Indexation step
indexer = get_block_indexer()

candidate_links = indexer.index(customer_doctors, alberta_doctors)
print('Potential candidates', len(candidate_links))
# Comparison step
comparer = get_comparer('qgram')
comparison_vectors = comparer.compute(candidate_links, customer_doctors, alberta_doctors)

# Classification step
scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[60, 30, 10])
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.82]
end_time = time.time()
duration = end_time - start_time

matches.to_csv('predictions_doctor.csv')
links_pred = pd.read_csv('predictions_doctor.csv', sep=',', index_col='Nr')

count_matches = len(links_pred)
count_nonmatches = len(customer_doctors) - len(links_pred)

save_results(links_pred.copy(), customer_doctors, alberta_doctors)

# Evaluation
print('Count matches', count_matches)
print('Count non-matches', count_nonmatches)
print('Duration', end_time - start_time)


