"""unitility method for read input file into list."""

import os
from config import ROOT_DIR


def load_training_data(file_name):
    """
    Read a file and convert into senteces list and label list
    Parameter: name of the input file
    return: list of 2D vector
    """
    with open(file=_get_path(file_name), mode="r", encoding="utf-8") as f:
        sentences = []
        labels = []
        for line in f:
            words = line.strip().split()
            if words:
                labels.append(words.pop())
            sentences.append(words)
    return sentences, labels


def _get_path(file_name):
    """
    Return absolute path to datasets/ folder from any calling location
    :param file_name: string file name
    :return absolute path
    """
    path = os.path.join(ROOT_DIR, "datasets", file_name)
    return os.path.abspath(path)
