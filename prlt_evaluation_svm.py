import recordlinkage as rl
import pandas
from recordlinkage.index import Full, Block
from recordlinkage.datasets import load_krebsregister

doctors_training = pandas.read_csv('Ärzte_Kunden_1000_SVM.csv',  sep=';',  skip_blank_lines=True)
all_training_pairs = Full().index(doctors_training)

matches_training_pairs = Block('Cluster').index(doctors_training)
comparer = rl.Compare()

training_vectors = comparer.compute(all_training_pairs, doctors_training)
# SVM
svm = rl.SVMClassifier()
svm.fit(training_vectors, matches_training_pairs)
svm_pairs = svm.predict(comparison_vectors)
svm_found_pairs_set = set(svm_pairs)

svm_true_positives = golden_pairs_set & svm_found_pairs_set
svm_false_positives = svm_found_pairs_set - golden_pairs_set
svm_false_negatives = golden_pairs_set - svm_found_pairs_set

print('true_positives total:', len(true_positives))
print('false_positives total:', len(false_positives))
print('false_negatives total:', len(false_negatives))
print()
print('svm_true_positives total:', len(svm_true_positives))
print('svm_false_positives total:', len(svm_false_positives))
print('svm_false_negatives total:', len(svm_false_negatives))