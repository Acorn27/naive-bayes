def print_confusion_matrix(nested_dict):
    """
    Prints a nested dictionary as a 2D table.
    :param nested_dict (dict): The nested dictionary to print.
    :returns: None
    """

    # Get all keys from the nested dictionary
    keys = set()
    for inner_dict in nested_dict.values():
        keys.update(inner_dict.keys())

    # Print table header
    print("Keys", end="\t")
    for key in keys:
        print(key, end="\t")
    print()

    # Print table rows
    for outer_key, inner_dict in nested_dict.items():
        print(outer_key, end="\t")
        for key in keys:
            value = inner_dict.get(key, "0")
            print(value, end="\t")
        print()
