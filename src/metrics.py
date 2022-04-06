import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics


# Label metric functions
def get_label_metrics(y_true, y_pred):
    """Returns dictionary of metrics calculated from labels."""
    cmatrix = get_confusion_matrix(y_true, y_pred)
    d = {'accuracy': get_accuracy(cmatrix),
         'sensitivity': get_sensitivity(cmatrix),
         'specificity': get_specificity(cmatrix),
         'precision': get_precision(cmatrix),
         'MCC': get_MCC(cmatrix),
         'F1': get_F1(cmatrix)}
    return d


def get_confusion_matrix(y_true, y_pred):
    """Returns a binary confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred are unequal lengths')
    tp, fp, tn, fn = 0, 0, 0, 0
    for label_true, label_pred in zip(y_true, y_pred):
        if label_true == 1:
            if label_pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if label_pred == 1:
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
    if any([s == 0 for s in (tp+fp, tp+fn, tn+fp, tn+fn)]):
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


# Score metric functions
def get_score_metrics(y_true, y_pred):
    """Returns dictionary of metrics calculated from scores."""
    d = {'ROC_AUC': get_ROC_AUC(y_true, y_pred),
         'PR_AUC': get_PR_AUC(y_true, y_pred),
         'cross_entropy': get_cross_entropy(y_true, y_pred)}
    return d


def get_ROC_AUC(y_true, y_pred):
    """Returns ROC AUC for scores."""
    return sklearn.metrics.roc_auc_score(y_true, y_pred)


def get_PR_AUC(y_true, y_pred):
    """Returns PR AUC for scores."""
    ps, rs, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    return sklearn.metrics.auc(rs, ps)


def get_cross_entropy(y_true, y_pred):
    """Returns cross entropy (log loss) for scores."""
    return sklearn.metrics.log_loss(y_true, y_pred)


# Plotting functions
def plot_ROC_curve(records, output_path):
    """Saves ROC curve."""
    for predictor, y_true, y_pred in records:
        fps, tps, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
        plt.plot(fps, tps, label=predictor)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.savefig(f'{output_path}/roc_plot.png')
    plt.close()


def plot_PR_curve(records, output_path):
    """Saves PR curve."""
    for predictor, y_true, y_pred in records:
        ps, rs, threshold = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        plt.plot(rs, ps, label=predictor)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{output_path}/pr_plot.png')
    plt.close()


# File IO
def load_labels(path, regex):
    """Return dictionary of labels as list keyed by accession."""
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
            if accession in records:
                raise RuntimeError(f'Duplicate accession detected: {accession}')
            elif not set(values) <= {0, 1}:
                raise RuntimeError(f'Non-binary values detected: {accession}')
            else:
                records[accession] = values
    return records


def load_scores(path, regex):
    """Return dictionary of scores as list keyed by accession."""
    records = {}
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith('>'):
                accession = re.search(regex, line).group(1)
                line = file.readline()

            values = []
            while line and not line.startswith('>'):
                values.append(float(line.rstrip()))
                line = file.readline()
            if accession in records:
                raise RuntimeError(f'Duplicate accession detected: {accession}')
            else:
                records[accession] = values
    return records


def check_accessions(y_true_accessions, y_pred_accessions, predictor='predictor'):
    """Raises error if y_pred_accessions is not a subset of y_true_accessions."""
    if not y_pred_accessions <= y_true_accessions:
        raise RuntimeError(
            f'y_pred accessions for {predictor} are not a subset of between y_true accessions and y_pred accessions')
    if y_pred_accessions != y_true_accessions:
        accessions = '\n'.join([accession for accession in (y_true_accessions - y_pred_accessions)])
        print(f'Warning: The following accessions are found in y_true but not y_pred for {predictor}:\n' + accessions)


def main(y_true_path, accession_regex,
         y_label_paths=None, y_score_paths=None, thresholds=None, visual=False, output_path='out/'):
    """Executes the full metrics pipeline.

    The pipeline will calculate metrics associated with both labels and scores
    for all predictors when possible. Label metrics are calculated from files
    given in y_label_paths. Labels can be computed on-the-fly from scores if
    thresholds are given. (If labels, scores, and thresholds are given, the
    pre-calculated labels are used where available.)

    Metrics are calculated with two strategies. In the target strategy, each
    metric is calculated for each protein individually, which are then
    averaged to yield the overall metric for each predictor. These results are
    saved as target.tsv and target_summary.tsv, respectively. Score metrics are
    not calculated in the target strategy. In the dataset strategy, all
    individual protein predictions are concatenated and treated as a single
    sequence from which label and score metrics are calculated.

    Parameters
    ----------
    y_true_path: str
        Path to true labels formatted as a FASTA-like  file where the headers
        are given as in FASTA files, but the sequences are replaced with
        binary (0, 1) labels indicating order and disorder, respectively
    accession_regex: str
        A regular expression to extract the accession from the header of each
        sequence in all files. The accession is extracted from the first group
        of the resulting match object, so it must be the first parenthesized
        subexpression.
    y_label_paths: 2-tuple of strings
        The first element is predictor label and second element is the path to
        the predicted labels. The predicted labels should be formatted as a
        FASTA-like file where the headers are given as in FASTA files, but the
        sequences are replaced with binary (0, 1) labels.
    y_score_paths: 2-tuple of strings
        The first element is predictor label and second element is the path to
        the predicted labels. The predicted labels should be formatted as a
        FASTA-like file where the headers are given as in FASTA files, but the
        score for each residue is given on a separate line.
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
    y_true_labels = load_labels(y_true_path, accession_regex)
    y_true_accessions = set(y_true_labels)

    # Load predicted labels
    predictor_labels = {}
    if y_label_paths is not None:
        for predictor, path in y_label_paths:
            y_pred_labels = load_labels(path, accession_regex)
            check_accessions(y_true_accessions, set(y_pred_labels), predictor)
            predictor_labels[predictor] = y_pred_labels
    if y_score_paths is not None and thresholds is not None:
        # Make thresholds dict
        if type(thresholds) == float:
            threshold = thresholds
            thresholds = {predictor: threshold for predictor, _ in y_score_paths}
        elif type(thresholds) != dict:
            raise RuntimeError('Thresholds is not float or dict')

        # Load scores and convert to labels
        for predictor, path in y_score_paths:
            if predictor in predictor_labels or predictor not in thresholds:
                continue  # Skip predictor if labels already loaded or threshold is not given
            threshold = thresholds[predictor]

            y_pred_scores = load_scores(path, accession_regex)
            y_pred_labels = {accession: [int(value >= threshold) for value in values] for accession, values in y_pred_scores.items()}
            check_accessions(y_true_accessions, set(y_pred_labels), predictor)
            predictor_labels[predictor] = y_pred_labels

    # Load predicted scores
    predictor_scores = {}
    if y_score_paths is not None:
        for predictor, path in y_score_paths:
            y_pred_scores = load_scores(path, accession_regex)
            check_accessions(y_true_accessions, set(y_pred_scores), predictor)
            predictor_scores[predictor] = y_pred_scores

    # Make output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Calculate label metrics of individual proteins (target strategy)
    if predictor_labels:
        rows = []
        for predictor, y_pred_labels in predictor_labels.items():
            for accession, y_pred in y_pred_labels.items():
                row = {'predictor': predictor, 'accession': accession}
                row.update(get_label_metrics(y_true_labels[accession], y_pred))
                rows.append(row)
        target = pd.DataFrame(rows)
        target_summary = target.groupby('predictor').mean()
        target.to_csv(f'{output_path}/target.tsv', sep='\t', index=False)
        target_summary.to_csv(f'{output_path}/target_summary.tsv', sep='\t')

    # Calculate label and score metrics aggregated by predictor (dataset strategy)
    # Label metrics
    if predictor_labels or predictor_scores:
        rows = []
        for predictor, y_pred_labels in predictor_labels.items():
            row = {'predictor': predictor}

            merge_pred, merge_true = [], []
            for accession, y_pred in y_pred_labels.items():
                merge_pred.extend(y_pred)
                merge_true.extend(y_true_labels[accession])
            row.update(get_label_metrics(merge_true, merge_pred))

            rows.append(row)
        labels = pd.DataFrame(rows)

        # Score metrics
        rows = []
        for predictor, y_pred_scores in predictor_scores.items():
            row = {'predictor': predictor}

            merge_pred, merge_true = [], []
            for accession, y_pred in y_pred_scores.items():
                merge_pred.extend(y_pred)
                merge_true.extend(y_true_labels[accession])
            row.update(get_score_metrics(merge_true, merge_pred))

            rows.append(row)
        scores = pd.DataFrame(rows)

        # Merge into single dataframe
        if not labels.empty and not scores.empty:
            dataset = labels.merge(scores, how='outer', on='predictor')
        else:
            dataset = labels if scores.empty else scores  # If both are empty, it doesn't matter which
        dataset.to_csv(f'{output_path}/dataset.tsv', sep='\t', index=False)

    # Plot visuals
    if predictor_scores and visual:
        records = []
        for predictor, y_pred_scores in predictor_scores.items():
            merge_true, merge_pred = [], []
            for accession, y_pred in y_pred_scores.items():
                merge_true.extend(y_true_labels[accession])
                merge_pred.extend(y_pred)
            records.append((predictor, merge_true, merge_pred))

        plot_PR_curve(records, output_path)
        plot_ROC_curve(records, output_path)
