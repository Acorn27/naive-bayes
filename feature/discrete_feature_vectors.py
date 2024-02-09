"""discrete feature vector implementation"""
from collections import defaultdict
from exceptions import UnknownWord


class DiscreteFeatureVector():
    """Discreate Feature Vector data structure"""

    def __init__(self, use_smoothing=False):
        """
        Construct a container for discrete feature vectors.
        :param use_smoothing: a boolean to indicate whether to use smoothing
        """
        self.use_laplace_smoothing = use_smoothing
        self.frequencies = defaultdict(lambda: defaultdict(int))
        # keep track of number of instance per label
        self.instance_count = defaultdict(int)
        # keep track of entire training set vocabulary
        self.vocabulary = set()

    def add(self, feature, label):
        """
        Update features constructor with new data from trainning set.
        :param feature: feature found during training process
        :param label: label for which feature belong to
        """
        self.frequencies[label][feature] += 1
        self.instance_count[label] += 1
        self.vocabulary.add(feature)

    def get_probability(self, feature, label):
        """
        Return probability of feature given label
        :param feature: a string feature/word we want to know probability
        :param label: a prior/context in which we want to compute probability
        """
        # check for unknown word
        if feature not in self.vocabulary:
            raise UnknownWord(feature)

        freq = self.frequencies[label][feature]
        num_words = self.instance_count[label]

        if self.use_laplace_smoothing:
            freq += 1
            num_words += len(self.vocabulary)
        return freq / num_words
