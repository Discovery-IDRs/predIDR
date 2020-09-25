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
    cmatrix = dict([("TP", 0),("TN", 0), ("FP",0), ("FN", 0) ])
    i = 0
    for s in seq:
        if seq[s] == 1:
            if ref[i] == 1:
                cmatrix["TP"] += 1
            else: #assume ref[i] == 0:
                cmatrix["FP"] += 1
        else: #assume s == 0
            if ref[i] == 1:
                cmatrix["FN"] += 1
            else: #assume ref[i] == 0:
               cmatrix["TN"] += 1
        i += 1
    return cmatrix



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
    i = 0
    cmatrix = dict([("TP", 0), ("TN", 0), ("FP", 0), ("FN", 0)])
    for s in seqlist:
        newcmatrix = get_confusion_matrix1(s, reflist[i])
        cmatrix["TP"] += newcmatrix["TP"]
        cmatrix["TN"] += newcmatrix["TN"]
        cmatrix["FP"] += newcmatrix["FP"]
        cmatrix["FN"] += newcmatrix["FN"]
        i+= 1
    return cmatrix

def get_accuracy(cmatrix):
    return (cmatrix["TP"] + cmatrix["TN"])/(cmatrix["TP"] + cmatrix["TN"]+ cmatrix["FP"] + cmatrix["FN"])

def get_MCC(cmatrix):
    return ((cmatrix["TP"] * cmatrix["TN"]) - (cmatrix["FP"] * cmatrix["FN"])) / math.sqrt((cmatrix["TP"] + cmatrix["FP"]) * (cmatrix["TP"] + cmatrix["FN"]) * (cmatrix["TN"] + cmatrix["FP"]) * (cmatrix["TN"] + cmatrix["FN"]))

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
    return (cmatrix["TP"]/(cmatrix["TP"] + cmatrix["FN"]))*100

def get_specificity(cmatrix):
    """Returns sensitivity for a 2x2 confusion matrix
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
    return (cmatrix["TN"]/(cmatrix["FP"] + cmatrix["TN"]))*100