from util import load_csv_training_data
import numpy as np
from cross_validation import k_fold_split, k_fold_iter
from evaluation import compute_precision, compute_recall
from evaluation import compute_macroaverage, print_confusion_matrix
from collections import defaultdict
from naive_bayes import NaiveBayes
k = 10
detail_print = False

def shuffle_data_and_labels(data, labels):
    # Generate shuffled indices
    num_samples = len(labels)
    shuffled_indices = np.random.permutation(num_samples)
    
    # Shuffle both data and labels based on the shuffled indices
    shuffled_data = [data[i] for i in shuffled_indices]
    shuffled_labels = [labels[i] for i in shuffled_indices]
    
    return shuffled_data, shuffled_labels


"""
Main function to load data, split it into training and testing sets,
and perform k-fold cross-validation.
:param input_file_name (str): Name of the file containing the dataset.
:returns: None
"""
label, data = load_csv_training_data("ecommerceDataset.csv")
shuffled_data, shuffled_label = shuffle_data_and_labels(data, label)

# Split training and label sets into k folds
segmented_datasets = k_fold_split(k, dataset=shuffled_data)
segmented_labelsets = k_fold_split(k, dataset=shuffled_label)

precision_dict = defaultdict(list)
recall_dict = defaultdict(list)
counter = 0

# iterate through all combination
for train_data, train_labels, test_data, test_labels in k_fold_iter(segmented_datasets, segmented_labelsets):

    # Train the model using the train set
    model = NaiveBayes(use_smoothing=True)
    model.fit(train_data, train_labels)

    # Evaluate the model on the test set
    confusion_matrix = model.evaluate(test_data, test_labels)

    # detailed print
    if detail_print:
        print(f"Fold #{counter+1}:\n")
        for i, test_instance in enumerate(test_data):
            print(f"{i + 1}. {' '.join(test_instance)}")
            print(f"\t-> Gold label: {test_labels[i]}")
            print(f"\t-> Model predict: {model.predict(test_instance)}")
        print("\nConfusion matrix:")
        print_confusion_matrix(confusion_matrix)
        print()

    for label in set(train_labels):
        precision = compute_precision(confusion_matrix, label)
        recall = compute_recall(confusion_matrix, label)
        precision_dict[label].append(precision)
        recall_dict[label].append(recall)

        if detail_print:
            print(
                f"""Precision({label})={float(precision)}, """,
                f"""Recall({label})={float(recall)}"""
            )
    counter += 1
    if detail_print:
        print("___________________________________________________________")

for label in set(train_labels):
    precision_macro = compute_macroaverage(precision_dict[label])
    recall_macro = compute_macroaverage(recall_dict[label])
    print(
        f"""Macroaveraged Precision({label})={float(precision_macro)}, """,
        f"""Macroaverage Recal({label})={float(recall_macro)}"""
    )
