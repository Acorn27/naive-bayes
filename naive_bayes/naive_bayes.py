"""Naive Bayes implemtation for sentiment analysis"""
from collections import defaultdict
from collections import Counter
from fractions import Fraction

from feature import DiscreteFeatureVector
from exceptions import UnknownWord


class NaiveBayes():
    """Implementation of naive Bayes."""
    def __init__(self, use_smoothing=False):
        """
        Construct a Naive Bayes classifier
        :param use_smoothing: whether to use laplace smoothing
        """

        # keep track of prior probibility for label/class
        self.priors = defaultdict(dict)
        self.use_smoothing = use_smoothing
        self.discrete_feature_vector = DiscreteFeatureVector(use_smoothing)
        self.is_trained = False
        self.label_count = Counter()

    def fit(self, training_sentences, labels_list):
        """
        Fit model according to training sentenes and correspoding label
        :param training_sentences: a list of string training sentences
        :label_list: a list of character label index-corresponding to sentences
        """
        for i, sentence in enumerate(training_sentences):
            label = labels_list[i]
            self.label_count[label] += 1
            for word in sentence:
                self.discrete_feature_vector.add(feature=word, label=label)

        num_records = len(labels_list)
        for label in self.label_count:
            self.priors[label] = Fraction(self.label_count[label], num_records)
        self.is_trained = True

    def predict(self, test_sentence, detail=False):
        """
        Predict label for a single test sentence
        :param test_sentence: string sentence
        """
        if not isinstance(test_sentence, list):
            test_sentence = test_sentence.split()

        if not self.is_trained:
            return False
        likelihood = {label: p_label for label, p_label in self.priors.items()}

        for label in self.label_count:

            if detail:
                print(f"P({label}) = {likelihood[label]}")

            for word in test_sentence:

                try:
                    p_word = self.discrete_feature_vector.get_probability(word, label)
                except UnknownWord as e:
                    if detail:
                        print(e)
                    continue

                if detail:
                    print(f'P({word}|{label}) = {p_word}')

                likelihood[label] *= p_word

            if detail:
                print(f"\t=> P({label}|d) = {likelihood[label]} = {float(likelihood[label])}\n")

        return max(likelihood, key=likelihood.get)

    def evaluate(self, test_data, test_labels):

        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for i, sentence in enumerate(test_data):
            predicted_label = self.predict(sentence)
            actual_label = test_labels[i]
            confusion_matrix[predicted_label][actual_label] += 1
        return confusion_matrix
