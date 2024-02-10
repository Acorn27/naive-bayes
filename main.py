"""Naives Bayes Model"""
import sys
from collections import defaultdict
from util import load_training_data
from naive_bayes import NaiveBayes
from evaluation import compute_precision
from evaluation import compute_recall
from evaluation import compute_macroaverage
from evaluation import print_confusion_matrix


def k_fold_split(k, dataset):
    """
    Splits the dataset into k parts.
    :param k (int): Number of parts to split the dataset into.
    :dataset (list): The dataset to be split.
    : returns: list: A list containing k segments of the dataset.
    """
    segment_length = len(dataset) // k
    split_dataset = [dataset[i * segment_length: (i + 1) * segment_length] for i in range(k)]
    return split_dataset


def main(input_file_name):
    """
    Main function to load data, split it into training and testing sets,
    and perform k-fold cross-validation.
    :param input_file_name (str): Name of the input file containing the dataset.
    :returns: None
    """
    # Load training data
    train_sets, label_sets = load_training_data(input_file_name)

    # Split training and label sets into k folds
    segmented_train_sets = k_fold_split(k=5, dataset=train_sets)
    segmented_label_sets = k_fold_split(k=5, dataset=label_sets)

    precision_dict = defaultdict(list)
    recall_dict = defaultdict(list)

    # Perform k-fold cross-validation
    for i in range(len(segmented_train_sets)):
        # Construct the training set by excluding the i-th fold
        train_data = [inner for outer in segmented_train_sets[:i]
                      + segmented_train_sets[i+1:] for inner in outer]
        train_labels = [inner for outer in segmented_label_sets[:i]
                        + segmented_label_sets[i+1:] for inner in outer]

        # Extract the i-th fold as the test set
        test_data = segmented_train_sets[i]
        test_labels = segmented_label_sets[i]

        # Train the model using the training data
        model = NaiveBayes(use_smoothing=True)
        model.fit(train_data, train_labels)

        # Evaluate the model on the test set
        confusion_matrix = model.evaluate(test_data, test_labels)

        # print(test_data)
        # print(test_labels)
        # print_confusion_matrix(confusion_matrix)

        for label in set(train_labels):
            precision_dict[label].append(compute_precision(confusion_matrix, label))
            recall_dict[label].append(compute_recall(confusion_matrix, label))

    precision_macro = compute_macroaverage(precision_dict["1"])
    recall_macro = compute_macroaverage(recall_dict["1"])

    print(f"model macro precision is {float(precision_macro)}")
    print(f"model macro recall is {float(recall_macro)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <program_name.py> <input_file.txt>")
    input_file_name = sys.argv[1]
    main(input_file_name)
