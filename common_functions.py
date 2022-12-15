import re
import pandas

def clean_name(row, remove, columnName):
    name_string = row[columnName]
    if isinstance(name_string,str):
        cleaned_name = re.sub(remove, '', name_string)
        return cleaned_name
    else:
        return ''

def save_results(links_pred, customer_nursingHomes, alberta_nursingHomes, entityName):
    pd_result = pandas.merge(customer_nursingHomes, links_pred, how="left", left_on='Nr', right_on='Nr')
    pd_result = pd_result.merge(alberta_nursingHomes, how='left', left_on='_id', right_index=True)
    pd_result = pd_result.rename(columns={'Name': 'Name Kunde', 'Adresse': 'Adresse Kunde', 'PLZ': 'PLZ Kunde', 'Ort': 'Ort Kunde', 
    'name': 'Name Alberta', 'address': 'Adresse Alberta', 'postalCode': 'PLZ Alberta', 'city': 'Ort Alberta'})
    #del pd_result["institution_name"]
    del pd_result["institution_address"]
    del pd_result["name_clean"]
    del pd_result["Name_clean"]
    del pd_result["score"]
    del pd_result["institution_postalcode"]

    pd_result.to_csv('results_matches_' + entityName + '.csv', encoding='utf-8')
  