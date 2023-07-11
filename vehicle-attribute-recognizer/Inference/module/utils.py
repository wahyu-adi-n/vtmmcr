from sklearn.metrics import classification_report, precision_recall_fscore_support, \
    precision_score, f1_score, recall_score, accuracy_score, \
    confusion_matrix
from module.class_names import class_names
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def set_seeds(seed:int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def label_encoding(class_names):
    return {key: i for i, key in enumerate(class_names)}


def inversion_label_encoding(label_enc):
    return {v: k for k, v in label_enc.items()}


def prec_score(y_true, y_pred, class_names):
    return precision_score(y_true, y_pred, labels=class_names, average='weighted')


def fone_score(y_true, y_pred, class_names):
    return f1_score(y_true, y_pred, labels=class_names, average='weighted')


def recc_score(y_true, y_pred, class_names):
    return recall_score(y_true, y_pred, labels=class_names, average='weighted')


def acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def classification_reports(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names)

def precision_recall_fscore_support(y_true, y_pred, average='weighted'):
    return precision_recall_fscore_support(y_true, y_pred, average=average)

def metrics_report_to_df(y_true, y_pred, dir_path):
    label_enc = label_encoding(class_names)
    inv_label_enc = inversion_label_encoding(label_enc)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred)
    classification_report = pd.concat(
        map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
    classification_report.columns = [
        "precision", "recall", "f1-score", "support"]  # Add row w "avg/total"
    classification_report.loc['avg/Total',
                              :] = precision_recall_fscore_support(y_true, y_pred, average='macro')
    classification_report.loc['avg/Total',
                              'support'] = classification_report['support'].sum()
    classification_report.rename(index=inv_label_enc, inplace=True)
    classification_report.to_csv(
        dir_path + f'classification_report.csv', encoding='utf-8')
    return classification_report


def save_plot_cm(y_true, y_pred, dir_path):
    plt.figure(figsize=(21, 14), dpi=400)
    sns.set(font_scale=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_matrix, annot=False)
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    figure = ax.get_figure()
    plt.show()
    figure.savefig(dir_path + f"confusion_matrix.png")
