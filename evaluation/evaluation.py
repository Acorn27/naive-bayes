from fractions import Fraction


def compute_precision(confusion_matrix, machine_label):

    correct_prediction = confusion_matrix.get(machine_label, {}).get(machine_label, 0)
    total_prediction = sum([gold_label for gold_label in confusion_matrix.get(machine_label, {}).values()])

    if correct_prediction == 0 or total_prediction == 0:
        return Fraction(0)
    return Fraction(correct_prediction, total_prediction)


def compute_recall(confusion_matrix, machine_label):

    correctly_identified = confusion_matrix.get(machine_label, {}).get(machine_label, 0)
    actually_present = sum([item.get(machine_label, 0) for item in confusion_matrix.values()])
    return Fraction(correctly_identified, actually_present)


def compute_F1(precision, recall, b):
    numerator = (1 + b ** 2) * precision * recall
    denominator = (b ** 2) * precision + recall
    return Fraction(numerator, denominator)


def compute_macroaverage(fractions):

    if not fractions:
        print('None')
        return None

    total = sum(fractions, Fraction(0))

    average = Fraction(total, len(fractions))

    return average
