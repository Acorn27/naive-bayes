"""unitility method for read input file into list."""

import os
import csv
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

def load_csv_training_data(file_name):
    with open(file=_get_path(file_name), mode="r", encoding="utf-8") as f:
        lines = csv.reader(f)
        dataset = list(lines)
        n = len(dataset)
        label = ["None" for i in range(n)]

        for i in range(len(dataset)):
            label[i] = dataset[i][0]
            dataset[i] = dataset[i][1:]
        return (label, dataset)


def _get_path(file_name):
    """
    Return absolute path to datasets/ folder from any calling location
    :param file_name: string file name
    :return absolute path
    """
    path = os.path.join(ROOT_DIR, "datasets", file_name)
    return os.path.abspath(path)
