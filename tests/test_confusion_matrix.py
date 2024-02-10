from fractions import Fraction
from pytest import approx
from evaluation import print_confusion_matrix
from evaluation import compute_precision
from evaluation import compute_recall
from evaluation import compute_macroaverage


def test_confusion_matrix():

    matrix = {
        "urgent": {"urgent": 8, "normal": 10, "spam": 1},
        "normal": {"urgent": 5, "normal": 60, "spam": 50},
        "spam": {"urgent": 3, "normal": 30, "spam": 200}
    }
    print_confusion_matrix(matrix)
    assert 1 == 1


def test_compute_precision():
    matrix = {
        "urgent": {"urgent": 8, "normal": 10, "spam": 1},
        "normal": {"urgent": 5, "normal": 60, "spam": 50},
        "spam": {"urgent": 3, "normal": 30, "spam": 200}
    }
    assert compute_precision(matrix, "urgent") == Fraction(8, 19)
    assert compute_precision(matrix, "normal") == Fraction(60, 115)
    assert compute_precision(matrix, "spam") == Fraction(200, 233)


def test_compute_precision_mod():
    matrix = {
        "urgent": {"urgent": 8, "normal": 10, "spam": 1},
        "normal": {"urgent": 5, "normal": 60, "spam": 50},
        "spam": {"urgent": 3, "normal": 30}
    }
    assert compute_precision(matrix, "urgent") == Fraction(8, 19)
    assert compute_precision(matrix, "normal") == Fraction(60, 115)
    assert compute_precision(matrix, "spam") == Fraction(0, 33)


def test_compute_precision_mod2():
    matrix = {
        "0": {"1": 1, "0": 1}
    }
    assert compute_precision(matrix, "0") == Fraction(1, 2)
    assert compute_precision(matrix, "1") == Fraction(0, 1)


def test_compute_recall():
    matrix = {
        "urgent": {"urgent": 8, "normal": 10, "spam": 1},
        "normal": {"urgent": 5, "normal": 60, "spam": 50},
        "spam": {"urgent": 3, "normal": 30, "spam": 200}
    }
    assert compute_recall(matrix, "urgent") == Fraction(8, 16)
    assert compute_recall(matrix, "normal") == Fraction(60, 100)
    assert compute_recall(matrix, "spam") == Fraction(200, 251)


def test_compute_recall_mod():
    matrix = {
        "urgent": {"urgent": 0, "normal": 0, "spam": 10},
        "normal": {"urgent": 5, "normal": 0, "spam": 0},
        "spam": {"urgent": 3, "normal": 30, "spam": 0}
    }
    assert compute_recall(matrix, "urgent") == Fraction(0, 8)
    assert compute_recall(matrix, "normal") == Fraction(0, 30)
    assert compute_recall(matrix, "spam") == Fraction(0, 10)


def test_average():
    fractions = [Fraction(8, 19), Fraction(60, 115), Fraction(200, 233)]
    assert round(float(compute_macroaverage(fractions)), 2) == approx(.60)
