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
    if len(seq) <= 1:
        return 0

    # Count terminal symbols
    repeat_count = 0
    if seq[0] == seq[1]:
        repeat_count += 1
    if seq[len(seq) - 1] == seq[len(seq) - 2]:
        repeat_count += 1

    # Count interior symbols
    if len(seq) > 2:
        for i in range(1, len(seq) - 1):
            if seq[i] == seq[i-1]:
                repeat_count += 1
            elif seq[i] == seq[i+1]:
                repeat_count += 1

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
    """Return count of pair symbols contained in XY in seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        XY : string or list
            Pair of symbols to count for repeats.
            Must contain at least two symbols.

    Returns
    -------
        pair_repeat_count : int
            Count of repeat pair symbols in seq.
    """
    if len(XY) < 2:
        raise ValueError('Requires at least two symbols.')
    if len(seq) <= 1:
        return 0

    # Count terminal symbols
    pair_repeat_count = 0
    if seq[0] in XY and seq[1] in XY:
        pair_repeat_count += 1
    if seq[len(seq) - 1] in XY and seq[len(seq) - 2] in XY:
        pair_repeat_count += 1

    # Count interior symbols
    if len(seq) > 2:
        for i in range(1, len(seq) - 1):
            if seq[i] in XY:
                if seq[i-1] in XY:
                    pair_repeat_count += 1
                elif seq[i+1] in XY:
                    pair_repeat_count += 1

    return pair_repeat_count


def get_pair_repeat_fractions(seq, XY, window_size):
    """Return fractions of pair symbols contained in XY in sliding
    window across seq which appear two or more times in a row.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        XY : string or list
            Pair symbols to count for repeats. Must contain at least
            two symbols.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        pair_repeat_fractions : list
            Fractions of pair symbols in sliding window across seq
            which appear two or more times in a row.
    """
    pair_repeat_fractions = []
    for i in range(len(seq)):
        window = get_window(seq, i, window_size)
        pair_repeat_count = get_pair_repeat_count(window, XY)
        pair_repeat_fractions.append(pair_repeat_count / len(window))
    return pair_repeat_fractions


def get_hydrophobicity(seq):
    """Return average hydrophobicity of symbols in seq.

    Parameters
    ----------
        seq : string
            Protein sequence as string.

    Returns
    -------
        hydrophobicity : int
            Score of hydrophobicity with most hydrophobic at 1 and
            most hydrophilic at 0.
    """
    hydrophobicity_dict = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
                           'M': 1.9, 'A': 1.8, 'W': -0.9, 'G': -0.4, 'T': -0.7,
                           'S': -0.8, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'N': -3.5,
                           'D': -3.5, 'Q': -3.5, 'E': -3.5, 'K': -3.9, 'R': -4.5}
    hydrophobicity_dict = {key: (value + 4.5) / 9 for key, value in hydrophobicity_dict.items()}
    seq_hydrophobicities = [hydrophobicity_dict.get(sym, 0) for sym in seq]
    return sum(seq_hydrophobicities) / len(seq_hydrophobicities)


def get_polarity(seq):
    """Return average polarity of symbols in seq.

    Parameters
    ----------
        seq : string
            Protein sequence as string.

    Returns
    -------
        polarity : int
            Score of average polarity ranging from 0 to 1, where 1 is
            polar and 0 is nonpolar.
    """
    polarity_dict = {'I': 0, 'V': 0, 'L': 0, 'F': 0, 'C': 0, 'M': 0, 'A': 0,
                     'W': 0, 'G': 0, 'T': 1, 'S': 1, 'Y': 1, 'P': 0, 'H': 1,
                     'N': 1, 'D': 1, 'Q': 1, 'E': 1, 'K': 1, 'R': 1}
    seq_polarities = [polarity_dict.get(sym, 0.5) for sym in seq]
    return sum(seq_polarities) / len(seq_polarities)


def get_content_scores(seq, content_func, window_size):
    """Return amino acid content score (scaled to 0-1) of amino acids in seq.

    Parameters
    ----------
        seq : string
            Protein sequence as string.
        content_func : function
            Gives content score for a specified sequence, ranging from 0 to 1.
        window_size : int
            Total number of symbols in window, including the center
            symbol. Must be an odd number.

    Returns
    -------
        content_scores : list
            Scores of content in sliding window across seq
    """
    content_scores = []
    for i in range(len(seq)):
        window = get_window(seq, i, window_size)
        score = content_func(window)
        content_scores.append(score)
    return content_scores


def AA_score(seq, ref, window_size):
    """Return list of average scores based on a reference dictionary."""
    scores = []
    for i in range(0, len(seq)):
        window = get_window(seq, i, window_size)
        window_scores = []
        for sym in window:
            if sym in ref:
                window_scores.append(ref[sym])
        scores.append(sum(window_scores) / len(window_scores))
    return scores


def get_features(seq, features, window_size):
    """Return values for each feature in features at each symbol in seq."""
    label2idx = {}  # Dictionary of (lists of feature values) keyed by feature label
    for feature_label, function in features.items():
        label2idx[feature_label] = function(seq, window_size)
    feature_labels = list(features)
    idx2label = []  # List of (dictionaries of feature values keyed by feature label)
    for i in range(len(seq)-1):
        idx2label.append({feature_label: label2idx[feature_label][i] for feature_label in feature_labels})
    return idx2label


# Reference dictionary for isoelectric point (average pKa)
AA_PI_value = {"G": 5.97, "A": 6.00, "V": 5.96, "L": 5.98, "I": 6.02, "M": 5.74, "P": 6.30, "F": 5.48,
               "W": 5.89, "N": 5.41, "Q": 5.65, "S": 5.68, "T": 5.60, "Y": 5.66, "C": 5.07, "D": 2.77,
               "E": 3.22, "K": 9.74, "R": 10.76, "H": 7.59}

# Reference dictionary of binary values based on the acidity/basicity of side chains
AA_acidic = {'D': 1, 'E': 1, 'K': 0, 'R': 0, 'H': 0, 'G': 0, 'A': 0, 'V': 0, 'I': 0, 'W': 0, 'F': 0, 'P': 0, 'M': 0,
             'L': 0, 'S': 0, 'T': 0, 'Y': 0, 'N': 0, 'Q': 0, 'C': 0}
AA_basic = {'D': 0, 'E': 0, 'K': 1, 'R': 1, 'H': 1, 'G': 0, 'A': 0, 'V': 0, 'I': 0, 'W': 0, 'F': 0, 'P': 0, 'M': 0,
            'L': 0, 'S': 0, 'T': 0, 'Y': 0, 'N': 0, 'Q': 0, 'C': 0}

# Feature functions and feature dictionary
features = {'hydrophobicity_content': lambda seq, window_size: get_content_scores(seq, get_hydrophobicity, window_size),
            'polarity_content': lambda seq, window_size: get_content_scores(seq, get_polarity, window_size),
            'aromatic_content': lambda seq, window_size: get_X_fractions(seq, 'FYWH', window_size),
            'acidic_content': lambda seq, window_size: AA_score(seq, AA_acidic, window_size),
            'basic_content': lambda seq, window_size: AA_score(seq, AA_basic, window_size),
            'isoelectric point': lambda seq, window_size: AA_score(seq, AA_PI_value, window_size),
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
            'fraction_AGP': lambda seq, window_size: get_X_fractions(seq, 'AGP', window_size),
            'fraction_SEG': lambda seq, window_size: get_X_fractions(seq, 'SEG', window_size)}
