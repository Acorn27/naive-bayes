class UnknownWord(ValueError):
    """ Raise inplace value error when words that
    occur in our test data but are not in our vocabulary at al"""
    def __init__(self, feature):
        """
        Constructor for UnknowWord exception
        :param feature: word that founded to be "unknown"
        """
        message = self.message(feature)
        super(UnknownWord, self).__init__(message)

    @staticmethod
    def message(feature):
        return (f"Unknown word caught '{str(feature)}' never occur in the training data. Skip")
