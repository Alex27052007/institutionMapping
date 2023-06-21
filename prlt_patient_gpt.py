import pandas as pd
import recordlinkage

# Lade die Daten
alberta_patients = pd.read_csv('./alberta/Alle_Patienten_Alberta.csv', sep=';', index_col='_id', dtype={'firstName': str, 'lastName': str, 'birthday': str})
customer_patients = pd.read_csv('./customer/patienten_sanikeil.csv', sep=';', index_col='acribaID', dtype={'Vorname': str, 'Nachname': str, 'GebDatum': str})

# Benenne die Spalten um
alberta_patients.rename(columns={'firstName': 'Vorname', 'lastName': 'Nachname', 'birthday': 'GebDatum'}, inplace=True)

# Bereinigung der Daten
alberta_patients['Vorname'] = alberta_patients['Vorname'].astype(str).str.lower()
alberta_patients['Nachname'] = alberta_patients['Nachname'].astype(str).str.lower()
alberta_patients['GebDatum'] = pd.to_datetime(alberta_patients['GebDatum'], format='%d.%m.%y', errors='coerce')

customer_patients['Vorname'] = customer_patients['Vorname'].astype(str).str.lower()
customer_patients['Nachname'] = customer_patients['Nachname'].astype(str).str.lower()
customer_patients['GebDatum'] = pd.to_datetime(customer_patients['GebDatum'], format='%d.%m.%y', errors='coerce')

# Überprüfe die Spaltennamen
print("Spalten in customer_patients:", customer_patients.columns)
print("Spalten in alberta_patients:", alberta_patients.columns)

# Indexierung und Linkage
indexer = recordlinkage.Index()
indexer.block('Nachname')
indexer.block('Vorname')
indexer.block('GebDatum')

candidate_links = indexer.index(customer_patients, alberta_patients)

compare_cl = recordlinkage.Compare()
compare_cl.string('Nachname', 'Nachname', method='qgram', threshold=0.8, label='last_name')
compare_cl.string('Vorname', 'Vorname', method='qgram', threshold=0.8, label='first_name')
compare_cl.exact('GebDatum', 'GebDatum', label='birthday')

comparison_vectors = compare_cl.compute(candidate_links, customer_patients, alberta_patients)

# Matching-Ergebnisse
matches = comparison_vectors[comparison_vectors.sum(axis=1) >= 2]

# Konvertiere die Index-Spalten zu regulären Spalten
matches.reset_index(inplace=True)
customer_patients.reset_index(inplace=True)
alberta_patients.reset_index(inplace=True)

# Füge alle Spalten aus den Originaltabellen hinzu
results = pd.merge(matches, customer_patients, left_on='acribaID', right_on='acribaID', how='left')
results = pd.merge(results, alberta_patients, left_on='_id', right_on='_id', how='left')

# Speichern der Ergebnisse
results.to_csv('results_matches_patient.csv', index=False)
