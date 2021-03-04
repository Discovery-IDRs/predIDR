import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import *


# Functions to calculate metrics for evaluating the performance of classifiers
def get_confusion_matrix(y_true, y_pred_bin):
    """Returns 2x2 confusion matrix for a single sequence."""
    check_inputs_valid(y_true, y_pred_bin)
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    return cm


def get_accuracy(y_true, y_pred_bin):
    """Returns accuracy for a 2x2 confusion matrix."""
    return accuracy_score(y_true, y_pred_bin)


def get_MCC(y_true, y_pred_bin):
    """Returns MCC (Matthews correlation coefficient) for a 2x2 confusion matrix."""
    cm = get_confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    if min(tp+fp, tp+fn, tn+fp, tn+fn) == 0:
        return 0
    return matthews_corrcoef(y_true, y_pred_bin)


def get_sensitivity(y_true, y_pred_bin):
    """Returns sensitivity for a 2x2 confusion matrix."""
    return recall_score(y_true, y_pred_bin, zero_division=0)


def get_specificity(y_true, y_pred_bin):
    """Returns specificity (true negative rate) for binary classification."""
    cm = get_confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)


def get_precision(y_true, y_pred_bin):
    """Returns precision for binary classification."""
    return precision_score(y_true, y_pred_bin, zero_division=0)


def get_f1(y_true, y_pred_bin):
    """Returns F1 score for binary classification."""
    return f1_score(y_true, y_pred_bin, zero_division=0)


def get_ROC_AUC(y_true, y_pred_dec):
    """Returns ROC AUC for decimal classification."""
    return roc_auc_score(y_true, y_pred_dec)


def get_PR_AUC(y_true, y_pred_dec):
    """Returns PR AUC for decimal classification."""
    p_array, r_array, threshold = precision_recall_curve(y_true, y_pred_dec)
    return auc(r_array, p_array)


def get_cross_entropy(y_true, y_pred_dec):
    """Returns cross entropy (log loss) for decimal predictions."""
    return log_loss(y_true, y_pred_dec)


# Functions to create visualizations
def get_ROC(y_true, y_pred_dec):
    """Saves ROC curve graph for decimal classification."""
    if not os.path.exists('out_metrics/'):
        os.mkdir('out_metrics/')

    fp_rate, tp_rate, threshold = roc_curve(y_true, y_pred_dec)
    plt.plot(fp_rate, tp_rate)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('out_metrics/roc_plot.png')
    plt.close()


def get_PR_curve(y_true, y_pred_dec):
    """Saves precision recall curve graph for decimal classification"""
    if not os.path.exists('out_metrics/'):
        os.mkdir('out_metrics/')

    p_array, r_array, threshold = precision_recall_curve(y_true, y_pred_dec)
    plt.plot(r_array, p_array)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('out_metrics/pr_plot.png')
    plt.close()


def get_cm_heatmap(y_true, y_pred_bin):
    """Saves confusion matrix heatmap for binary classification."""
    if not os.path.exists('out_metrics/'):
        os.mkdir('out_metrics/')

    cm = get_confusion_matrix(y_true, y_pred_bin)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 16})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('out_metrics/cm_plot.png')
    plt.close()


# Functions to check inputs
def check_inputs_valid(y_true, y_pred_bin):
    """Returns validity of input lists or raises Exception.

       Parameters
       ----------
        y_true : list
            Actual labels for each residue, ordered as in the original
            sequence.
        y_pred_bin : list
            Predicted labels for each residue, ordered as in the original
            sequence. Assumes 0 == not disordered, 1 == disordered.

       Returns
       -------
           isValid : bool
             Returns True if y_true and y_pred_bin inputs are non-empty, the
             same length, and only contain binary values.
    """
    if len(y_true) == 0:
        raise ValueError('y_true must be non-empty')
    if len(y_pred_bin) == 0:
        raise ValueError('y_pred_bin must be non-empty')
    if len(y_true) != len(y_pred_bin):
        raise ValueError('y_true and y_pred_bin must be the same length')
    if not check_binary(y_true):
        raise ValueError('y_true must only contain 1s and 0s')
    if not check_binary(y_pred_bin):
        raise ValueError('y_pred_bin must only contain 1s and 0s')
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
    p = set(list(vals))
    s = {0, 1}
    if s == p or p == {0} or p == {1}:
        return True
    else:
        return False


# Pipeline: Metrics and Visualizations
def get_metrics(y_true, y_pred, visual=False):
    """Return dataframe of all possible metrics given predicted classification."""
    if check_binary(y_pred):
        binary_metrics_df = get_binary_metrics(y_true, y_pred)
        decimal_metrics_df = pd.DataFrame({'ROC_AUC': [None],
                                           'PR_AUC': [None],
                                           'Cross Entropy': [None]})
        if visual:
            get_cm_heatmap(y_true, y_pred)
    else:
        y_pred_bin = [round(val) for val in y_pred]
        binary_metrics_df = get_binary_metrics(y_true, y_pred_bin)
        decimal_metrics_df  = get_decimal_metrics(y_true, y_pred)
        if visual:
            get_visualizations(y_true, y_pred_bin, y_pred)
    metrics_df = pd.concat([binary_metrics_df, decimal_metrics_df], axis=1)
    return metrics_df


def get_binary_metrics(y_true, y_pred_bin):
    """Return dataframe of metrics with binary classification."""
    d = {'Accuracy': get_accuracy(y_true, y_pred_bin),
         'MCC': get_MCC(y_true, y_pred_bin),
         'Sensitivity': get_sensitivity(y_true, y_pred_bin),
         'Specificity': get_specificity(y_true, y_pred_bin),
         'Precision': get_precision(y_true, y_pred_bin),
         'F1': get_f1(y_true, y_pred_bin)}
    binary_metrics_df = pd.DataFrame(data=d, index=[0])
    return binary_metrics_df


def get_decimal_metrics(y_true, y_pred_dec):
    """Return dataframe of metrics with decimal classification."""
    d = {'ROC_AUC': get_ROC_AUC(y_true, y_pred_dec),
         'PR_AUC': get_PR_AUC(y_true, y_pred_dec),
         'Cross Entropy': get_cross_entropy(y_true, y_pred_dec)}
    decimal_metrics_df = pd.DataFrame(data=d, index=[0])
    return decimal_metrics_df


def get_visualizations(y_true, y_pred_bin, y_pred_dec):
    """Return visualizations for binary and decimal classifications."""
    get_ROC(y_true, y_pred_dec)
    get_PR_curve(y_true, y_pred_dec)
    get_cm_heatmap(y_true, y_pred_bin)
    return

def get_all_metrics(y_true, y_pred_list, visual_list=[], predictors):
    """Return dataframe of all possible metrics given multiple predictors."""
    df_list =[]
    for i in range(len(y_pred_list)):
        df_list.append(get_metrics(y_true, y_pred_list[i], visual_list[i]))
    
    output_df = pd.concat(df_list, ignore_index=True)
    output_df['Predictor'] = predictors
    output_df.set_index('Predictor')
    return output_df

def get_all_metrics_by_protein(y_true_dict, y_pred_dict_list, visual_list=[], predictors):
    df_list = []
    if len(y_pred_dict_list) == 1:
        for y_pred_dict in y_pred_dict_list:
            for key in y_pred_dict:
                y_true = y_true_dict[key]
                y_pred = y_pred_dict[key]
                df_list.append(get_metrics(get_metrics(y_true, y_pred))
            
            output_df = pd.concat(df_list, ignore_index=True)
            output_df['DISPROT_ID'] = y_pred_dict.keys()
            output_df.set_index('DISPROT_ID')
    else:
        for y_pred_dict in y_pred_dict_list:
            for key in y_pred_dict:
                y_true = y_true_dict[key]
                y_pred = y_pred_dict[key]
                df = get_metrics(get_metrics(y_true, y_pred))
                df['DISPROT_ID'] = y_pred_dict.keys()
                df_list.append(df)
            output_df = pd.concat(df_list, ignore_index=True)
        output_df['Predictor'] = predictors
        output_df.set_index('Predictor')
    
    return output_df

# File IO
def get_rid_newline(fasta_file):
    """input name of fasta file as string and function returns string 
    containing contents of fasta formatted without newlines in sequences"""
    
    with open(fasta_file) as file:
        data = file.read()
    
    lines = data.splitlines()
    
    new_lines_str = ""
    
    for line in lines:
        if line[0] == ">":
            new_lines_str = new_lines_str + "\n" + line + "\n"
        elif line[0] == "0":
            new_lines_str = new_lines_str + line
        elif line[0] == "1":
            new_lines_str = new_lines_str + line
        else:
            pass
    
    new_lines_str = new_lines_str[1:]
    
    return new_lines_str

def fasta_file_to_dict(fasta_file):
    """input name of fasta file as string and function converts that fasta file into dict"""
    
    clean_fasta_str = get_rid_newline(fasta_file)
        
    lines = clean_fasta_str.splitlines()
    
    disprot_id = []
    sequences = []
    
    for line in lines:
        if line[0] == ">":
            disprot_id.append(line[12:19])
        else:
            int_list = []
            for i in line:
                int_list.append(int(i))
            sequences.append(int_list)
            
    return dict(zip(disprot_id, sequences))

def get_y_true_dict(y_true_fasta_file="disprot_2020_06.fasta"):
    y_true_dict = fasta_to_dict(y_true_fasta_file)
    return y_true_dict

def get_y_pred_dict(y_pred_fasta_file):
    y_pred_dict = fasta_to_dict(y_pred_fasta_file)
    return y_pred_dict

def get_y_true(y_true_dict): 
    for key in y_true_dict:
        y_true.append(y_true_dict[key])
    return y_true

def get_y_pred(y_pred_dict):
    for key in y_pred_dict:
        y_pred.append(y_pred_dict[key])
    return y_pred

def main(y_pred_fasta_files, y_true_fasta_file, by_protein=False, visual_list=[], all_visual=False):
    if not os.path.exists('out_metrics/'):
        os.mkdir('out_metrics/')
    
    if not (all_visual or visual_list):
        visual_list =[False for i in range(y_true_list) ]
    elif all_visual and not visual_list :
        visual_list =[True for i in range(y_true_list) ] 
    
    y_true_dict = get_y_true_dict(y_pred_fasta_files)
    y_true = get_y_pred(y_pred_fasta_file)
    
    y_pred_list = []
    y_pred_dict_list =[]
    predictors = [] 
    for y_pred_fasta_file in y_pred_fasta_files:
        if by_protein:
            y_pred_dict = get_y_pred_dict(y_pred_fasta_file)
            y_pred_dict_list.append(y_pred_dict)
        y_pred_list.append(get_y_pred(y_pred_dict))
        predictor = y_pred_fasta_file.split(".")[0]
        predictors.append(predictor)

    if by_protein:
        if any(visual_list):
            print("Visuals by protein not available. Please run without by_protein flag for visuals. Metrics still computed.")
            df = get_all_metrics_by_protein(y_true, y_pred_list, [False for i in range(y_true_list)], predictors)
            df.to_csv(r'out_metrics/predictor_metrics_byprotein.csv', index=False)
    else:
        df = get_all_metrics(y_true_dict, y_pred_dict_list, visual_list, predictors)
        df.to_csv(r'out_metrics/predictor_metrics.csv', index=False)
    return



