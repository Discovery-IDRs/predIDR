import math

"""Functions to calculate metrics for evaluating the performance of classifers."""


def get_confusion_matrix1(seq, ref):
    """Return 2x2 confusion matrix for a single sequence.

    Assume inputs are well-formed: (perhaps method to do this check is necessary)
    -seq and ref are same length
    -both contain only 0 and 1: 0 represents not disordered, 1 represents disorded
    -non-null or empty inputs

    Parameters
    ----------
        seq : list
            Predicted labels for each residue, ordered as in the original
            sequence. Let's assume 0 = not disordered, 1 = disordered.
        ref : list
            Actual labels for each residue, ordered as in the original sequence.

    Returns
    -------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.
    """

    TP, FP, TN, FN = 0, 0, 0, 0

    ref_counter = 0
    for s in seq:
        r = ref[ref_counter]
        if s == 0:
            if r == s:
                TN += 1
            else:
                FN += 1
        elif s == 1:
            if r == s:
                TP += 1
            else:
                FP += 1
        ref_counter += 1

    conf_mat_dict = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }

    return conf_mat_dict


def get_confusion_matrix2(seqlist, reflist):
    """Return 2x2 confusion matrix for a list of sequences.

    Parameters
    ----------
        seqlist : list of lists
            List of predicted labels for each residue, ordered as in the
            original sequence.
        reflist : list of lists
            List of actual labels for each residue, ordered as in the original
            sequence.

    Returns
    -------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.
    """

    TP, FP, TN, FN = 0, 0, 0, 0
    ref_counter = 0

    for s in seqlist:
        r = reflist[ref_counter]
        dict = get_confusion_matrix1(s, r)
        TP += dict.get("TP")
        FP += dict.get("FP")
        TN += dict.get("TN")
        FN += dict.get("FN")
        ref_counter += 1

    conf_mat_dict = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }

    return conf_mat_dict


def get_accuracy(cmatrix):
    """Returns accuracy for a 2x2 confusion matrix.

    Parameters
    ----------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.

    Returns
    -------
        accuracy : float
            Accuracy (frequency of correct classifications) from the cmatrix
    """
    return (cmatrix['TP'] + cmatrix['TN']) / sum(cmatrix.values())


def get_MCC(cmatrix):
    """Returns MCC (Matthews correlation coefficient) for a 2x2 confusion matrix

    Parameters
    ----------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.

    Returns
    -------
        mcc : float
            MCC (decimal between -1 and +1) indicating the quality of binary
            classifications from cmatrix
    """
    tp, tn, fp, fn = cmatrix['TP'], cmatrix['TN'], cmatrix['FP'], cmatrix['FN']
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1
    return numerator / (denominator ** 0.5)


def get_sensitivity(cmatrix):
    """Returns sensitivity for a 2x2 confusion matrix
       Parameters
       ----------
           cmatrix : dict
               Counts for true positives, true negatives, false positives, and
               false negatives, keyed by TP, TN, FP, and FN, respectively.
       Returns
       -------
           sensitivity : float
               Measures the proportion of positives that are correctly identified
    """
    return (cmatrix["TP"] / (cmatrix["TP"] + cmatrix["FN"]))


def get_specificity(cmatrix):
    """Returns specificity for a 2x2 confusion matrix
       Parameters
       ----------
           cmatrix : dict
               Counts for true positives, true negatives, false positives, and
               false negatives, keyed by TP, TN, FP, and FN, respectively.
       Returns
       -------
           specificity : float
               Measures the proportion of negatives that are correctly identified
    """
    return (cmatrix["TN"] / (cmatrix["FP"] + cmatrix["TN"]))


def get_precision(cmatrix):
    """Returns precision for a 2x2 confusion matrix
       Parameters
       ----------
           cmatrix : dict
               Counts for true positives, true negatives, false positives, and
               false negatives, keyed by TP, TN, FP, and FN, respectively.
       Returns
       -------
           precision : float
               Measures how many selected items are relevant
    """
    return (cmatrix["TP"] / (cmatrix["TP"] + cmatrix["FP"]))


def get_f1(cmatrix, b):
    """Returns F1 score for a 2x2 confusion matrix
       Parameters
       ----------
           cmatrix : dict
               Counts for true positives, true negatives, false positives, and
               false negatives, keyed by TP, TN, FP, and FN, respectively.
           b : float
               A postive, real factor such that recall (sensitivity) is
               considered b times as important as precision
       Returns
       -------
           f1 : float
               Harmonic mean of precision of recall
               Value between 0 and 1
               Measure of a test's accuracy
    """

    precision = get_precision(cmatrix)
    recall = get_sensitivity(cmatrix)
    return (1 + b ** 2) * ((precision * recall) / (b ** 2 * precision + recall))


def check_inputs_valid(seq, ref):
    """Returns validity of input lists or raises Exception
       Parameters
       ----------
        seq : list
            Predicted labels for each residue, ordered as in the original
            sequence. Let's assume 0 = not disordered, 1 = disordered.
        ref : list
            Actual labels for each residue, ordered as in the original sequence.
       Returns
       -------
           isValid : bool
             Returns True if seq and ref inputs are non-empty, the same length, and only contain binary values
    """
    if not seq:
        raise Exception('Seq must be non-empty')
    if not ref:
        raise Exception('Ref must be non-empty')
    if len(seq) != len(ref):
        raise Exception('Seq and ref must be the same length')
    if not check_binary(seq):
        raise Exception('Seq must only contain 1s and 0s')
    if not check_binary(ref):
        raise Exception('Ref must only contain 1s and 0s')
    return True


def check_binary(vals):
    """Returns whether all values in a list are binary
       Parameters
       ----------
        vals: list
            labels for each residue
       Returns
       -------
           isBinary : bool
             Returns True if vals only contain binary values, False otherwise
    """
    p = set(vals)
    s = {'0', '1'}
    if s == p or p == {'0'} or p == {'1'}:
        return True
    else:
        return False
