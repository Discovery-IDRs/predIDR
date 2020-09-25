"""Functions to calculate metrics for evaluating the performance of classifers."""


def get_confusion_matrix1(seq, ref):
    """Return 2x2 confusion matrix for a single sequence.

    Parameters
    ----------
        seq : list
            Predicted labels for each residue, ordered as in the original
            sequence.
        ref : list
            Actual labels for each residue, ordered as in the original sequence.

    Returns
    -------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for s in seq:
        for r in ref:
            if s == 0:
                if r == s:
                    TN += 1
                else:
                    FP += 1
            elif s == 1:
                if r == s:
                    TP += 1
                else:
                    FN += 1

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

    for s in seqlist:
        for r in reflist:
            dict = get_confusion_matrix1(s,r)
            TP += dict.get("TP")
            FP += dict.get("FP")
            TN += dict.get("TN")
            FN += dict.get("FN")

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
    pass


def get_specificity(cmatrix):
    pass
