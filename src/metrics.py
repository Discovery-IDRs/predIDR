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
    tp, fp, tn, fn = 0, 0, 0, 0

    for s in seq:
        for r in ref:
            if s == 0:
                if r == s:
                    tn += 1
                else:
                    fp += 1
            elif s == 1:
                if r == s:
                    tp += 1
                else:
                    fn += 1

    conf_mat_dict = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
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
    tp, fp, tn, fn = 0, 0, 0, 0

    for s in seqlist:
        for r in reflist:
            dict = get_confusion_matrix1(s,r)
            tp += dict.get("tp")
            fp += dict.get("fp")
            tn += dict.get("tn")
            fn += dict.get("fn")

    conf_mat_dict = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

    return conf_mat_dict


def get_accuracy(cmatrix):
    pass


def get_MCC(cmatrix):
    pass


def get_sensitivity(cmatrix):
    pass


def get_specificity(cmatrix):
    pass
