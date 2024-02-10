def k_fold_split(k: int, dataset):
    """
    Splits the dataset into k parts.
    :param k (int): Number of parts to split the dataset into.
    :dataset (list): The dataset to be split.
    : returns: list: A list containing k segments of the dataset.
    """
    segment_length = len(dataset) // k
    return [dataset[i * segment_length: (i + 1) * segment_length]
            for i in range(k)]


def k_fold_iter(segmented_datas, segmented_labels):
    """
    Iterate over k folds of the dataset.
    :param k (int): Number of folds.
    :param datasets (list): List of k segments of the dataset.
    :returns: generator: A generator yielding tuples of
    (train_data, train_labels, test_data, test_labels)
    """
    for i in range(len(segmented_datas)):
        train_data = [inner for outer in
                      segmented_datas[:i] + segmented_datas[i + 1:]
                      for inner in outer]
        train_labels = [inner for outer in
                        segmented_labels[:i] + segmented_labels[i + 1:]
                        for inner in outer]
        test_data = segmented_datas[i]
        test_labels = segmented_labels[i]
        yield train_data, train_labels, test_data, test_labels
