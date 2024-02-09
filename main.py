"""Naives Bayes Model"""
import sys
from util import load_training_data
from naive_bayes import NaiveBayes


def main(input_file_name):
    """ Main function"""
    sentences, labels = load_training_data(input_file_name)

    my_model = NaiveBayes(use_smoothing=True)

    my_model.fit(training_sentences=sentences, labels_list=labels)

    c = my_model.predict("gently embraces blue ripples", detail=True)

    print(f"Predicted class is {c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python <program_name.py> <input_file.txt>")
    input_file_name = sys.argv[1]
    main(input_file_name)
