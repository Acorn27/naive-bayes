"""Call Naives Bayes"""
from util import load_training_data
from naive_bayes import NaiveBayes


def main():
    """ Main function"""
    sentences, labels = load_training_data('dataset1.txt')

    my_model = NaiveBayes(use_smoothing=True)

    my_model.fit(training_sentences=sentences, labels_list=labels)

    c = my_model.predict("Chinese Chinese Chinese Tokyo Japan")

    print(c)


if __name__ == "__main__":
    main()
