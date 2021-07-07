import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics


# Binary metric functions
def get_confusion_matrix(y_true, y_pred):
    """Returns a binary confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred are unequal lengths')
    tp, fp, tn, fn = 0, 0, 0, 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1:
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == 1:
                fp += 1
            else:
                tn += 1
    cmatrix = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    return cmatrix


def get_accuracy(cmatrix):
    """Return accuracy for a 2x2 confusion matrix."""
    s = sum(cmatrix.values())
    if s == 0:
        return np.nan
    return (cmatrix['tp'] + cmatrix['tn']) / s


def get_F1(cmatrix, b=1):
    """Returns F1 score for a 2x2 confusion matrix."""
    precision = get_precision(cmatrix)
    recall = get_sensitivity(cmatrix)
    denominator = b ** 2 * precision + recall
    if denominator == 0:
        return np.nan
    return (1 + b ** 2) * ((precision * recall) / denominator)


def get_MCC(cmatrix):
    """Returns MCC (Matthews correlation coefficient) for a 2x2 confusion matrix."""
    tp, tn, fp, fn = cmatrix['tp'], cmatrix['tn'], cmatrix['fp'], cmatrix['fn']
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if min(tp+fp, tp+fn, tn+fp, tn+fn) == 0:
        return np.nan
    return numerator / (denominator ** 0.5)


def get_precision(cmatrix):
    """Returns precision for a 2x2 confusion matrix."""
    tp, fp = cmatrix['tp'], cmatrix['fp']
    if tp + fp == 0:
        return np.nan
    return tp / (tp + fp)


def get_sensitivity(cmatrix):
    """Returns sensitivity for a 2x2 confusion matrix."""
    tp, tn, fp, fn = cmatrix['tp'], cmatrix['tn'], cmatrix['fp'], cmatrix['fn']
    if tp + fn == 0:
        return np.nan
    return tp / (tp + fn)


def get_specificity(cmatrix):
    """Returns sensitivity for a 2x2 confusion matrix."""
    tn, fp = cmatrix['tn'], cmatrix['fp']
    if tn + fp == 0:
        return np.nan
    return tn / (tn + fp)


# Decimal metric functions
def get_ROC_AUC(y_true, y_pred):
    """Returns ROC AUC for decimal classification."""
    return sklearn.metrics.roc_auc_score(y_true, y_pred)


def get_PR_AUC(y_true, y_pred):
    """Returns PR AUC for decimal classification."""
    p_array, r_array, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    return sklearn.metrics.auc(r_array, p_array)


def get_cross_entropy(y_true, y_pred):
    """Returns cross entropy (log loss) for decimal predictions."""
    return sklearn.metrics.log_loss(y_true, y_pred)


# Plotting functions
def plot_ROC_curve(y_true, y_pred, output_path):
    """Saves ROC curve graph for decimal classification."""
    fp_rate, tp_rate, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
    plt.plot(fp_rate, tp_rate)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'{output_path}/roc_plot.png')
    plt.close()


def plot_PR_curve(y_true, y_pred, output_path):
    """Saves precision recall curve graph for decimal classification."""
    p_array, r_array, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    plt.plot(r_array, p_array)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'{output_path}/pr_plot.png')
    plt.close()


# Validation functions
def is_binary(labels):
    """Returns whether all values in a list are binary.

       Parameters
       ----------
        labels: iterable
            Labels for each residue. Can be given as strings or ints (or even a
            mixture of the two).

       Returns
       -------
           bool
             Returns True if labels only contain binary values, False otherwise.
    """
    for label in labels:
        if label not in [0, 1]:
            return False
    return True


def is_binary_file(path):
    with open(path) as file:
        file.readline()
        try:
            return set(file.readline().rstrip()) == set('01')
        except ValueError:
            return False


# Pipeline: Metrics and Visualizations
def get_binary_metrics(y_true, y_pred):
    """Returns dictionary of metrics with binary labels."""
    cmatrix = get_confusion_matrix(y_true, y_pred)
    d = {'accuracy': get_accuracy(cmatrix),
         'MCC': get_MCC(cmatrix),
         'sensitivity': get_sensitivity(cmatrix),
         'specificity': get_specificity(cmatrix),
         'precision': get_precision(cmatrix),
         'F1': get_F1(cmatrix)}
    return d


def get_decimal_metrics(y_true, y_pred):
    """Returns dictionary of metrics with decimal scores."""
    d = {'ROC_AUC': get_ROC_AUC(y_true, y_pred),
         'PR_AUC': get_PR_AUC(y_true, y_pred),
         'cross_entropy': get_cross_entropy(y_true, y_pred)}
    return d


def get_visualizations(y_true, y_pred):
    """Plots visualizations for binary and decimal classifications."""
    plot_ROC_curve(y_true, y_pred)
    plot_PR_curve(y_true, y_pred)


# File IO
def load_binary(path, regex):
    records = {}
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                accession = re.search(regex, line).group(1)
                line = file.readline()

            values = []
            while line and not line.startswith('>'):
                values.extend([int(value) for value in line.rstrip()])
                line = file.readline()
            if accession not in records:
                records[accession] = values
            else:
                raise RuntimeError(f'Duplicate accession detected: {accession}')
    return records


def load_decimal(path, regex):
    records = {}
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                accession = re.search(regex, line).group(1)
                line = file.readline()

            values = []
            while line and not line.startswith('>'):
                sym, value = line.split()
                values.append(float(value))
                line = file.readline()
            if accession not in records:
                records[accession] = value
            else:
                raise RuntimeError(f'Duplicate accession detected: {accession}')
    return records


def main(y_true_path, y_pred_paths, accession_regex, thresholds=0.5, visual=False, output_path='out/'):
    """Executes the full metrics pipeline.

    The pipeline has two somewhat distinct behaviors depending on whether the
    predicted labels are in binary or decimal format. (This is automatically
    detected.) If the labels are binary, only binary metrics are computed. If
    the labels are decimal, binary and decimal metrics are computed. The binary
    labels are calculated from the decimal scores by applying a threshold above
    which the residue is labeled disordered.

    Parameters
    ----------
    y_true_path: str
        Path to true labels formatted as a FASTA-like binary where 1 and 0
        indicate disorder and order, respectively.
    y_pred_paths: 2-tuple of strings
        The first element is predictor label and second element is the path to
        the predicted labels. The predicted labels should either be formatted
        as a FASTA-like binary or FASTA-like decimal file where the headers
        are given as in FASTA files, but the score for each residue is given on
         a separate line. These scores must be in [0, 1].
    accession_regex: str
        A regular expression to extract the accession from the header of each
        sequence in all files. The accession is extracted from the first group
        of the resulting match object, so it must be the first parenthesized
        subexpression.
    thresholds: float or dict
        If decimal value is greater than or equal to threshold, the residue is
        labeled as disordered for the calculation of binary metrics. If float,
        threshold for converting decimal predictions is set globally. Otherwise
        the thresholds are set individually, stored in a dict keyed by the
        predictor label given in y_pred_paths.
    visual: bool
        If true, output includes plots.
    output_path: str
        Path to output directory. The directly will be created if it does not
        exist.

    Returns
    -------
        No return value (i.e. None); output is written to output_path.
    """
    # Load true labels
    y_true_records = load_binary(y_true_path, accession_regex)
    for accession, labels in y_true_records.items():
        if not is_binary(labels):
            raise RuntimeError(f'Non-binary y_true labels detected: {accession}')
    accessions = set(y_true_records)

    # Make thresholds dict
    predictors = set([predictor for predictor, _ in y_pred_paths])
    if type(thresholds) == float:
        default = thresholds
        thresholds = {}
        for predictor, _ in y_pred_paths:
            thresholds[predictor] = default
    elif type(thresholds) == dict:
        if set(thresholds) != predictors:
            raise RuntimeError('Thresholds do not match predictors in y_pred_paths')
    else:
        raise RuntimeError('Thresholds is not float or dict')

    # Load predicted labels
    y_pred_binaries, y_pred_decimals = {}, {}
    for predictor, y_pred_path in y_pred_paths:
        binary_bool = is_binary_file(y_pred_path)
        if binary_bool:
            y_pred_binary = load_binary(y_pred_path, accession_regex)

            for accession, labels in y_pred_binary.items():
                if not is_binary(labels):
                    raise RuntimeError(f'Non-binary y_pred labels detected in {predictor}: {accession}')
        else:
            y_pred_decimal = load_decimal(y_pred_path, accession_regex)
            y_pred_decimals[predictor] = y_pred_decimal

            threshold = thresholds[predictor]
            y_pred_binary = {}
            for accession, values in y_pred_decimal.items():
                y_pred_binary[accession] = [1 if value >= threshold else 0 for value in values]

        if set(y_pred_binary) != accessions:
            raise RuntimeError(f'Mismatch between y_true accessions and y_pred accessions in {predictor}')
        y_pred_binaries[predictor] = y_pred_binary

    # Make output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Calculate binary metrics of individual proteins
    rows = []
    for predictor, y_pred_records in y_pred_binaries.items():
        for accession, y_pred in y_pred_records.items():
            row = {'predictor': predictor, 'accession': accession}
            row.update(get_binary_metrics(y_true_records[accession], y_pred))
            rows.append(row)
    target = pd.DataFrame(rows)
    target_summary = target.groupby('predictor').mean()
    target.to_csv(f'{output_path}/target.tsv', sep='\t', index=False)
    target_summary.to_csv(f'{output_path}/target_summary.tsv', sep='\t')

    # Calculate binary and decimal metrics at the level of proteins
    rows = []
    for predictor in predictors:
        y_pred_binary = y_pred_binaries[predictor]
        merge = []
