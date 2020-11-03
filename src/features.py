"""Functions to calculate features associated with each amino acid in a protein sequence."""


def get_window(seq, pos, window_size):
    """Return window of length window_size centered at position pos in seq.

    If the window exceeds the bounds of the seq, get_window returns
    the maximal possible window. Thus, the window at the upper and
    lower bounds are actually right- and left- facing half windows,
    respectively.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        pos : int
            Index of the center position of the window.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        window : string
            Window of length window_size centered at position pos in
            seq.
    """
    if pos < 0 or pos > len(seq) - 1:
        raise ValueError('Pos is outside the bounds of seq.')
    if window_size % 2 == 0:
        raise ValueError('Size is even.')
    delta = (window_size - 1) // 2

    lower = pos - delta
    if lower < 0:
        lower = 0
    upper = pos + delta + 1  # Add 1 to include upper bound
    if upper > len(seq):
        upper = len(seq)
    return seq[lower:upper]


def get_X_count(seq, X):
    """Return count of symbols in X in seq.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        X : string or list
            Symbols to count as string or list.

    Returns
    -------
        X_count : int
            Count of symbols in X in seq.
    """
    X_count = 0
    for sym in seq:
        if sym in X:
            X_count += 1
    return X_count


def get_X_fractions(seq, X, window_size):
    """Return fractions of symbols in X in sliding window across seq.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        X : string or list
            Symbols to count as string or list.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        X_fractions : list
            Fractions of symbols in X in sliding window across seq
    """
    X_fractions = []
    for i in range(len(seq)):
        window = get_window(seq, i, window_size)
        X_count = get_X_count(window, X)
        X_fractions.append(X_count / len(window))
    return X_fractions


def get_regex_count(seq, regex):
    """Return count of patterns matching regex in seq."""
    pass


def get_regex_fractions(seq, regex, window):
    """Return fractions of patterns matching X in sliding window across seq."""
    pass


# TODO: Not all features will necessarily accept a window_size parameter, so we may need to rework this
def get_features(seq, features, window_size):
    """Return values for each feature in features at each symbol in seq."""
    label2idx = {}  # Dictionary of (lists of feature values) keyed by feature label
    for feature_label, function in features.items():
        label2idx[feature_label] = function(seq, window_size)
    feature_labels = list(features)
    idx2label = []  # List of (dictionaries of feature values keyed by feature label)
    for i in range(len(seq)):
        idx2label.append({feature_label: label2idx[feature_label][i] for feature_label in feature_labels})
    return idx2label


features = {'fraction_AGP': lambda seq, window_size: get_X_fractions(seq, 'AGP', window_size),
            'fraction_?': lambda seq, window_size: get_X_fractions(seq, '?', window_size)}
