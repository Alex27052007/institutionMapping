import recordlinkage
from recordlinkage.preprocessing import clean
import pandas
from recordlinkage.base import BaseCompareFeature
import re
import numpy as np
from common_functions import clean_name 


def save_results(links_pred, customer_nursingServices, alberta_nursingServices):
    pd_result = pandas.merge(customer_nursingServices, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_nursingServices, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'Adresse': 'Adresse Kunde', 'PLZ': 'PLZ Kunde', 'Ort': 'Ort Kunde', 
    'name': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    #del pd_result["institution_name"]
    del pd_result["institution_address"]
    del pd_result["Name_clean"]

    pd_result.to_csv('results_matches_nursingservice.csv', encoding='utf-8')
  
   
def get_block_indexer():
   indexer = recordlinkage.Index()
   indexer.block(left_on='Adresse', right_on='address')
   indexer.block(left_on='PLZ', right_on='postalCode')
   return indexer

def get_sorted_neighbourhood_indexer ():
   indexer = recordlinkage.SortedNeighbourhoodIndex(
   left_on='PLZ', right_on='postalCode', window=9
   )
   return indexer
def get_comparer(methodName):
    comparer = recordlinkage.Compare()
    #comparer.string('Name_clean', 'name_clean', method=methodName, label='institution_name')
    comparer.string('Adresse', 'address', method=methodName, label='institution_address')
    comparer.exact('PLZ', 'postalCode', label='institution_postalcode')

    return comparer

alberta_nursingServices = pandas.read_csv('alberta/Alle_Pflegedienste_Alberta.csv',  sep=';', index_col='_id', dtype={ 'postalCode': str,})
customer_nursingServices = pandas.read_csv('customer/nursingService_mwm.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})

#Cleaning
parts_to_remove = '\.|\.|gGmbH|GmbH|Diakonie|Caritas|Altenpflegeheim|Seniorenzentrum|Caritas|Pflegeheim|Seniorenheim|Senioren- und Pflegezentrum|Seniorenhaus|Seniorenpflegeheim|AWO|ASB|Pflegezentrum|BRK|Altenheim|Seniorenwohnheim|Seniorenhaus|Seniorendomizil'
customer_nursingServices['Name_clean'] = customer_nursingServices.apply(lambda row: clean_name(row, parts_to_remove, 'Name'), axis=1)
alberta_nursingServices['name_clean'] = alberta_nursingServices.apply(lambda row: clean_name(row, parts_to_remove, 'name'),axis=1)

# Indexation step
indexer = get_block_indexer()

candidate_links = indexer.index(customer_nursingServices, alberta_nursingServices)
print('Potential candidates', len(candidate_links))
# Comparison step
comparer = get_comparer('qgram')
comparison_vectors = comparer.compute(candidate_links, customer_nursingServices, alberta_nursingServices)

# Classification step
scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[60, 40])
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.68]

matches.to_csv('predictions_ns.csv')
links_pred = pandas.read_csv('predictions_ns.csv', sep=',', index_col='Nr')

count_matches = len(links_pred)
count_nonmatches = len(customer_nursingServices) - len(links_pred)

save_results(links_pred.copy(), customer_nursingServices, alberta_nursingServices)

# Evaluation
print('Count matches', count_matches)
print('Count non-matches', count_nonmatches)


