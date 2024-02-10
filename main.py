""" Custom Naives Bayes Model implement from scratch """
import argparse
from collections import defaultdict
from util import load_training_data
from naive_bayes import NaiveBayes
from evaluation import compute_precision, compute_recall
from evaluation import compute_macroaverage, print_confusion_matrix
from cross_validation import k_fold_split, k_fold_iter


def cross_validation(input_file_name, k=1, detail_print=False):
    """
    Main function to load data, split it into training and testing sets,
    and perform k-fold cross-validation.
    :param input_file_name (str): Name of the file containing the dataset.
    :returns: None
    """
    # Load training data
    data_sets, label_sets = load_training_data(input_file_name)

    # Split training and label sets into k folds
    segmented_datasets = k_fold_split(k, dataset=data_sets)
    segmented_labelsets = k_fold_split(k, dataset=label_sets)

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


def prediction(input_file, test_sentence, detail_print):

    # Load training data
    data_sets, label_sets = load_training_data(input_file)

    # Train the model using the train set
    model = NaiveBayes(use_smoothing=True)
    model.fit(data_sets, label_sets)

    predicted_label = model.predict(test_sentence=test_sentence,
                                    detail=detail_print)

    print(f"Model predicted label is: {predicted_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Naives Bayes Model")
    parser.add_argument("input_file", help="Input file name")
    parser.add_argument("test_sentence", nargs='?', default='', help="String of test sentence")
    parser.add_argument("-rP", action="store_true", help="Set detail print to True")

    args = parser.parse_args()

    input_file = args.input_file
    test_sentence = args.test_sentence
    set_detail_print = args.rP

    if not test_sentence:
        cross_validation(input_file, k=5, detail_print=set_detail_print)
    else:
        prediction(input_file, test_sentence, detail_print=set_detail_print)
