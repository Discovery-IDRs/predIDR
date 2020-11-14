"""Functions to calculate features associated with each amino acid in a protein sequence."""


# Helper functions
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

def get_repeat_count(seq):
    """Return count of symbols in seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.

    Returns
    -------
        repeat_count : int
            Count of repeat symbols in seq.
    """
    repeat_count = 0

    if len(seq) <= 1:
        return 0
    else:
        pass

    if seq[0] == seq[1]:
        repeat_count += 1
    else:
        pass

    if seq[len(seq) - 1] == seq[len(seq) - 2]:
        repeat_count += 1
    else:
        pass

    if len(seq) > 2:
        for i in range(1, len(seq) - 1):
            if seq[i] == seq[i-1]:
                repeat_count += 1
            elif seq[i] == seq[i+1]:
                repeat_count += 1
            else:
                pass
    else:
        pass

    return repeat_count

def get_repeat_fractions(seq, window_size):
    """Return fractions of symbols in sliding window across seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        repeat_fractions : list
            Fractions of symbols in sliding window across seq which appear two or more times in a row.
    """

    repeat_fractions = []

    for i in range(len(seq)):
        window = get_window(seq, i, window_size)
        repeat_count = get_repeat_count(window)
        repeat_fractions.append(repeat_count / len(window))

    return repeat_fractions

def get_pair_repeat_count(seq, XY):
    """Return count of pair symbols contained in XY
    in seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        XY : string or list
            Pair symbols to count for repeats.
            Must contain at least two symbols.

    Returns
    -------
        pair_repeat_count : int
            Count of repeat pair symbols in seq.
    """

    if len(XY) < 2:
        raise ValueError('Requires at least two symbols.')

    pair_repeat_count = 0

    if len(seq) <= 1:
        return 0
    else:
        pass

    if seq[0] in XY:
        if seq[1] in XY:
            pair_repeat_count += 1
        else:
            pass
    else:
        pass

    if seq[len(seq) - 1] in XY:
        if seq[len(seq) - 2] in XY:
            pair_repeat_count += 1
        else:
            pass
    else:
        pass

    if len(seq) > 2:
        for i in range(1, len(seq) - 1):
            if seq[i] in XY:
                if seq[i-1] in XY:
                    pair_repeat_count += 1
                elif seq[i+1] in XY:
                    pair_repeat_count += 1
                else:
                    pass
            else:
                pass
    else:
        pass

    return pair_repeat_count

def get_pair_repeat_fractions(seq, XY, window_size):
    """Return fractions of pair symbols contained in XY in sliding
    window across seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        XY : string or list
            Pair symbols to count for repeats.
            Must contain at least two symbols.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        pair_repeat_fractions : list
            Fractions of pair symbols in sliding window across
            seq which appear two or more times in a row.
    """

    pair_repeat_fractions = []

    for i in range(len(seq)):
        window = get_window(seq, i, window_size)
        pair_repeat_count = get_pair_repeat_count(window, XY)
        pair_repeat_fractions.append(pair_repeat_count / len(window))

    return pair_repeat_fractions

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


# Feature functions and feature dictionary
features = {'fraction_AGP': lambda seq, window_size: get_X_fractions(seq, 'AGP', window_size),
            'fraction_SEG': lambda seq, window_size: get_X_fractions(seq, 'SEG', window_size),
            'fraction_pos_charge': lambda seq, window_size: get_X_fractions(seq, 'KRH', window_size),
            'fraction_neg_charge': lambda seq, window_size: get_X_fractions(seq, 'DE', window_size),
            'fraction_repeat': lambda seq, window_size: get_repeat_fractions(seq, window_size),
            'fraction_QN_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'QN', window_size),
            'fraction_RG_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'RG', window_size),
            'fraction_FG_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'FG', window_size),
            'fraction_SG_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'SG', window_size),
            'fraction_SR_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'SR', window_size),
            'fraction_KAP_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'KAP', window_size),
            'fraction_PTS_repeat': lambda seq, window_size: get_pair_repeat_fractions(seq, 'PTS', window_size),
            'fraction_G': lambda seq, window_size: get_X_fractions(seq, 'G', window_size),
            'fraction_A': lambda seq, window_size: get_X_fractions(seq, 'A', window_size),
            'fraction_V': lambda seq, window_size: get_X_fractions(seq, 'V', window_size),
            'fraction_L': lambda seq, window_size: get_X_fractions(seq, 'L', window_size),
            'fraction_I': lambda seq, window_size: get_X_fractions(seq, 'I', window_size),
            'fraction_M': lambda seq, window_size: get_X_fractions(seq, 'M', window_size),
            'fraction_F': lambda seq, window_size: get_X_fractions(seq, 'F', window_size),
            'fraction_W': lambda seq, window_size: get_X_fractions(seq, 'W', window_size),
            'fraction_P': lambda seq, window_size: get_X_fractions(seq, 'P', window_size),
            'fraction_S': lambda seq, window_size: get_X_fractions(seq, 'S', window_size),
            'fraction_T': lambda seq, window_size: get_X_fractions(seq, 'T', window_size),
            'fraction_C': lambda seq, window_size: get_X_fractions(seq, 'C', window_size),
            'fraction_Y': lambda seq, window_size: get_X_fractions(seq, 'Y', window_size),
            'fraction_N': lambda seq, window_size: get_X_fractions(seq, 'N', window_size),
            'fraction_Q': lambda seq, window_size: get_X_fractions(seq, 'Q', window_size),
            'fraction_D': lambda seq, window_size: get_X_fractions(seq, 'D', window_size),
            'fraction_E': lambda seq, window_size: get_X_fractions(seq, 'E', window_size),
            'fraction_K': lambda seq, window_size: get_X_fractions(seq, 'K', window_size),
            'fraction_R': lambda seq, window_size: get_X_fractions(seq, 'R', window_size),
            'fraction_H': lambda seq, window_size: get_X_fractions(seq, 'H', window_size),
            'fraction_?': lambda seq, window_size: get_X_fractions(seq, '?', window_size)}
