import recordlinkage
from recordlinkage.preprocessing import clean
import pandas
from recordlinkage.base import BaseCompareFeature
import re
import numpy as np
from common_functions import clean_name 


def save_results(links_pred, customer_nursingHomes, alberta_nursingHomes):
    pd_result = pandas.merge(customer_nursingHomes, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_nursingHomes, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'Adresse': 'Adresse Kunde', 'PLZ': 'PLZ Kunde', 'Ort': 'Ort Kunde', 
    'name': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    #del pd_result["institution_name"]
    del pd_result["institution_address"]
    del pd_result["Name_clean"]

    pd_result.to_csv('results_matches_nursinghome.csv', encoding='utf-8')
  
   
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

def evaluate (methodname, weights, duration, method_results, predicted_links, true_links):
    true_positives = 0
    false_positives = 0
    false_positives_none = 0
    true_negatives = 0
    false_negatives = 0

    for index, predicted_link in predicted_links.iterrows():
        predicted_match_id = predicted_link.loc['_id']
        real_match_id = links_true.loc[index, '_id']
        
        if real_match_id == 'None':
            false_positives_none += 1
        
        if predicted_match_id == real_match_id:
            true_positives += 1
        else:
            false_positives += 1

    true_negatives = count_real_non_matches - false_positives_none
    false_negatives = count_nonmatches - true_negatives
    if (true_positives == 0 and false_positives == 0):
        precision = 0
        recall = 0
        fscore = 0
    else:
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        fscore = 2 * precision * recall/(precision + recall)
    method_results.append([method, weights, duration, true_positives, false_positives, false_negatives, true_negatives, precision, recall, fscore])
    print('true_positives', true_positives)
    print('false_positives', false_positives)
    print('true_negatives', true_negatives)
    print('false_negatives', false_negatives)
    print('precision', precision)
    print('recall', recall)
    print('fscore', fscore)   


alberta_nursingHomes = pandas.read_csv('alberta/Alle_Pflegeheime_Alberta.csv',  sep=';', index_col='_id', dtype={ 'postalCode': str,})
customer_nursingHomes = pandas.read_csv('customer/nursingHome_mwm.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})

#Cleaning
parts_to_remove = '\.|\.|gGmbH|GmbH|Diakonie|Caritas|Altenpflegeheim|Seniorenzentrum|Caritas|Pflegeheim|Seniorenheim|Senioren- und Pflegezentrum|Seniorenhaus|Seniorenpflegeheim|AWO|ASB|Pflegezentrum|BRK|Altenheim|Seniorenwohnheim|Seniorenhaus|Seniorendomizil'
customer_nursingHomes['Name_clean'] = customer_nursingHomes.apply(lambda row: clean_name(row, parts_to_remove, 'Name'), axis=1)
alberta_nursingHomes['name_clean'] = alberta_nursingHomes.apply(lambda row: clean_name(row, parts_to_remove, 'name'), axis=1)

# Indexation step
indexer = get_block_indexer()

candidate_links = indexer.index(customer_nursingHomes, alberta_nursingHomes)
print('Potential candidates', len(candidate_links))
# Comparison step
comparer = get_comparer('qgram')
comparison_vectors = comparer.compute(candidate_links, customer_nursingHomes, alberta_nursingHomes)

# Classification step
scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[60, 40])
scored_comparison_vectors = comparison_vectors.assign(score=scores)

matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.68]

matches.to_csv('predictions_nh.csv')
links_pred = pandas.read_csv('predictions_nh.csv', sep=',', index_col='Nr')

count_matches = len(links_pred)
count_nonmatches = len(customer_nursingHomes) - len(links_pred)

save_results(links_pred.copy(), customer_nursingHomes, alberta_nursingHomes)

# Evaluation
print('Count matches', count_matches)
print('Count non-matches', count_nonmatches)


