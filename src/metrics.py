import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import *

"""Functions to calculate metrics for evaluating the performance of classifiers."""


def get_confusion_matrix1(seq, ref):
    """Returns 2x2 confusion matrix for a single sequence.

    Parameters
    ----------
        seq : list
            Predicted labels for each residue, ordered as in the original
            sequence. Assumes 0 == not disordered, 1 == disordered.
        ref : list
            Actual labels for each residue, ordered as in the original sequence.

    Returns
    -------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.
    """
    check_inputs_valid(seq, ref)

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

    cmatrix = {"TP": TP,
               "FP": FP,
               "TN": TN,
               "FN": FN}

    return cmatrix


def get_confusion_matrix2(seqlist, reflist):
    """Returns 2x2 confusion matrix for a list of sequences.

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
        check_inputs_valid(s, r)
        dict = get_confusion_matrix1(s, r)
        TP += dict.get("TP")
        FP += dict.get("FP")
        TN += dict.get("TN")
        FN += dict.get("FN")
        ref_counter += 1

    cmatrix = {"TP": TP,
               "FP": FP,
               "TN": TN,
               "FN": FN}

    return cmatrix


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
            Accuracy (frequency of correct classifications) from the cmatrix.
    """
    return (cmatrix["TP"] + cmatrix["TN"]) / sum(cmatrix.values())


def get_MCC(cmatrix):
    """Returns MCC (Matthews correlation coefficient) for a 2x2 confusion matrix.

    Parameters
    ----------
        cmatrix : dict
            Counts for true positives, true negatives, false positives, and
            false negatives, keyed by TP, TN, FP, and FN, respectively.

    Returns
    -------
        mcc : float
            MCC (decimal between -1 and +1) indicating the quality of binary
            classifications from cmatrix.
    """
    TP, TN, FP, FN = cmatrix["TP"], cmatrix["TN"], cmatrix["FP"], cmatrix["FN"]
    numerator = TP * TN - FP * FN
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) or 1
    if denominator == 0:
        return float("NaN")
    return numerator / (denominator ** 0.5)


def get_sensitivity(cmatrix):
    """Returns sensitivity for a 2x2 confusion matrix.

       Parameters
       ----------
           cmatrix : dict
               Counts for true positives, true negatives, false positives, and
               false negatives, keyed by TP, TN, FP, and FN, respectively.

       Returns
       -------
           sensitivity : float
               Measures the proportion of positives that are correctly identified.
    """
    numerator = cmatrix["TP"]
    denominator = (cmatrix["TP"] + cmatrix["FN"])
    if denominator == 0:
        return float("NaN")
    return numerator/denominator


def get_specificity(y_true, y_pred_bin):
    """Returns specificity (true negative rate) for binary classification."""
    cm = confusion_matrix(y_true, y_pred_bin) # exchange for our cm func later
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    return tn / (tn + fp)


def get_precision(y_true, y_pred_bin):
    """Returns precision for binary classification."""
    return precision_score(y_true, y_pred_bin, zero_division=0)


def get_f1(y_true, y_pred_bin):
    """Returns f1 score for binary classification."""
    return f1_score(y_true, y_pred_bin, zero_division=0)


def get_AUC(y_true, y_pred_dec):
    """Returns AUC for decimal classification."""
    return roc_auc_score(y_true, y_pred_dec)


def get_cross_entropy(y_true, y_pred_dec):
    """Returns cross entropy (log loss) for decimal predictions."""
    return log_loss(y_true, y_pred_dec)


"""Functions to create visualizations"""


def get_ROC(y_true, y_pred_dec):
    """Saves ROC curve graph for decimal classification."""
    if not os.path.exists('out_metrics_rewritten/'):
        os.mkdir('out_metrics_rewritten/')

    fp_rate, tp_rate, threshold = roc_curve(y_true, y_pred_dec)
    plt.plot(fp_rate, tp_rate)
    plt.title('ROC Curve');
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('out_metrics_rewritten/roc_plot.png')
    plt.close()


def get_cm_heatmap(y_true, y_pred_bin):
    """Saves confusion matrix heatmap for binary classification."""
    if not os.path.exists('out_metrics_rewritten/'):
        os.mkdir('out_metrics_rewritten/')

    cm = confusion_matrix(y_true, y_pred_bin)
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual');
    plt.savefig('out_metrics_rewritten/cm_plot.png')
    plt.close()


"""Helper Input-Checking Functions"""


def check_inputs_valid(seq, ref):
    """Returns validity of input lists or raises Exception.

       Parameters
       ----------
        seq : list
            Predicted labels for each residue, ordered as in the original
            sequence. Assumes 0 == not disordered, 1 == disordered.
        ref : list
            Actual labels for each residue, ordered as in the original sequence.

       Returns
       -------
           isValid : bool
             Returns True if seq and ref inputs are non-empty, the same length, and only contain binary values.
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
    """Returns whether all values in a list are binary.

       Parameters
       ----------
        vals: list
            Labels for each residue.

       Returns
       -------
           isBinary : bool
             Returns True if vals only contain binary values, False otherwise.
    """
    p = set(vals)
    s = {0, 1}
    if s == p or p == {0} or p == {1}:
        return True
    else:
        return False


"""Draft of Pipeline"""


def get_metrics(y_true, y_pred, visual=False):
    """Return all possible metrics given predicted classification."""
    if check_binary(y_pred):
        get_binary_metrics(y_true, y_pred)
    else:
        y_pred_bin = [round(val) for val in y_pred]
        get_binary_metrics(y_true, y_pred_bin)
        get_decimal_metrics(y_true, y_pred)
        if visual:
            get_visualizations(y_true, y_pred_bin, y_pred)

def get_binary_metrics(y_true, y_pred_bin):
    """Return metrics with binary classification."""
    return

def get_decimal_metrics()y_true, y_pred:
    """Return metrics with decimal classification."""
    return

def get_visualizations(y_true, y_pred_bin, y_pred_dec):
    """Return visualizations for binary and decimal classifications"""
    return
