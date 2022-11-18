import math
import numpy as np
import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean
from recordlinkage.base import BaseCompareFeature
from recordlinkage.index import Full, Block
import re
import time



parts_to_remove = '\.|\.|Herr|Frau|Dr|Drs|med|medic|univ|Dipl|Prof|Priv|Doz|Praxis|Gemeinschaftspraxis|GP|rer|soc|Facharzt|Klinik|Kardiologie|Urologie|Schlaflabor|Hausarzt'

def clean_name(row):
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
    'fullName': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    del pd_result["institution_name"]
    del pd_result["institution_address"]
    del pd_result["firstName"]
    del pd_result["lastName"]
    del pd_result["Name_clean"]

    pd_result.to_csv('results_matches_doctor.csv')

def get_full_indexer():
    indexer = recordlinkage.Index()
    indexer.add(Full())
    return indexer
   
def get_block_indexer():
    indexer = recordlinkage.Index()
    indexer.block(left_on='PLZ', right_on='postalCode')
    indexer.block(left_on='Name', right_on='fullName')
    return indexer

def get_sorted_neighbourhood_indexer ():
   indexer = recordlinkage.SortedNeighbourhoodIndex(
   left_on='PLZ', right_on='postalCode', window=9
   )
   return indexer

def get_comparer(methodName):
    comparer = recordlinkage.Compare()
    comparer.string('Name_clean', 'fullName', method=methodName, label='institution_name')
    comparer.string('PLZ', 'postalCode', method=methodName, label='institution_postalCode')
    comparer.string('Adresse', 'address', method=methodName, label='institution_address')


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

alberta_doctors = pd.read_csv('Alle_Ärzte_Alberta.csv',  sep=';', index_col='_id', dtype={'gender': str, 'postalCode': str,})
customer_doctors = pd.read_csv('Ärzte_Kunden_1000.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})
links_true = pd.read_csv('Ärzte_Kunden_1000_True_Links.csv',  sep=';', index_col='Nr', dtype={'Nr': str, 'PLZ': str})

#Cleaning
customer_doctors['Name_clean'] = customer_doctors.apply(clean_name, axis=1)
count_real_non_matches = 50

methods = ['jarowinkler', 'qgram', 'smith_waterman', 'lcs']

#probable_weights=[[65, 35], [60, 40], [55, 45], [50, 50], [45, 55], [40, 60], [35, 65]]
probable_weights=[[50, 30, 20], [60, 30, 10], [45, 10, 45]]
all_methods_results = []

for method in methods:
    for probable_weight in probable_weights:
        start_time = time.time()
        # Indexation step
        indexer = get_sorted_neighbourhood_indexer()

        candidate_links = indexer.index(customer_doctors, alberta_doctors)
        print('Potential candidates', len(candidate_links))
        # Comparison step
        comparer = get_comparer(method)
        comparison_vectors = comparer.compute(candidate_links, customer_doctors, alberta_doctors)
        # Classification step
        scores = np.average(
            comparison_vectors.values,
            axis=1,
            weights=probable_weight)
        scored_comparison_vectors = comparison_vectors.assign(score=scores)
        matches = scored_comparison_vectors[scored_comparison_vectors['score'] >= 0.85]
        end_time = time.time()
        duration = end_time - start_time

        matches.to_csv('predictions.csv')
        links_pred = pd.read_csv('predictions.csv', sep=',', index_col='Nr')

        count_matches = len(links_pred)
        count_nonmatches = len(customer_doctors) - len(links_pred)

        save_results(links_pred.copy(), customer_doctors, alberta_doctors)

        # Evaluation step
        evaluate(method, probable_weight, duration, all_methods_results, links_pred, links_true)

df = pd.DataFrame(all_methods_results, columns=['Methode', 'Gewichtung', 'Dauer', 'TP', 'FP', 'FN', 'TN', 'Prec', 'Recall', 'Fscore'])
df.to_csv('Gesamtergebnis.csv')



