import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from utils import prepend_string_to_array
from tabulate import tabulate


def evaluate(y_test, y_pred, multilabelbinarizer, classes):
    """
    Evaluates the results.

    Computes the precision, recall and F1-score for each of the classes.

    Parameters
    ----------
    y_test : list
        The list of the true data.
    y_pred : list
        The list of the predicted data.
    multilabelbinarizer : sklearn.preprocessing.MultiLabelBinarizer
        The binarizer that maps each class to its target vector.
    classes : list(str)
        The list of classes to be evaluated.

    Returns
    -------
    precision : list(float)
        The precision for each of the evaluated classes.
        precision = tp / (tp + fp)
    recall : list(float)
        The recall for each of the evaluated classes.
        recall = tp / (tp + fn)
    f1_score : list(float)
        The F1-score for each of the evaluate classes.
        F1 = 2 * precision * recall / (precision + recall)
    support : list(int)
        Number of tested samples for each class.
    """

    p, r, f1, s = precision_recall_fscore_support(y_test, y_pred)

    precision = []
    recall = []
    f1_score = []
    support = []

    for l in classes:
        i = np.where(multilabelbinarizer.transform([(l, )]) == 1)[1][0]

        f1_score.append(f1[i])
        precision.append(p[i])
        recall.append(r[i])
        support.append(s[i])

    return precision, recall, f1_score, support


def evaluate_single_run(classes, precisions, recalls, f1_scores, supports):
    """
    Wrapper for evaluate for pretty printing.


    Parameters
    ----------
    classes : list(str)
        A list of which classes that should be evaluated.
    precisions : numpy.ndarray
        Precisions for each of the run.
    recalls
    f1_scores
    supports

    Returns
    -------

    """
    results = {}
    headers = ['F1', 'Precision', 'Recall', 'Support']
    for i, c in enumerate(classes):
        results[c] = [f1_scores[i], precisions[i], recalls[i], supports[i]]

    return results, headers


def evaluate_multiple_runs(classes, precisions, recalls, f1_scores, supports):
    """
    Wrapper for evaluate for multiple runs.

    Calculates the mean and standard deviation for Precision, Recall, F1 and number of samples.

    Parameters
    ----------
    classes : list(str)
        A list of which classes that should be evaluated.
    precisions : numpy.ndarray
        Precisions for each of the run.
    recalls
    f1_scores
    supports

    Returns
    -------

    """
    precision_means = np.mean(precisions, axis=0)
    precision_stds = np.std(precisions, axis=0)

    recalls_means = np.mean(recalls, axis=0)
    recalls_stds = np.std(recalls, axis=0)

    f1_scores_means = np.mean(f1_scores, axis=0)
    f1_scores_stds = np.std(f1_scores, axis=0)

    supports_means = np.mean(supports, axis=0)
    support_stds = np.std(supports, axis=0)

    results = {}

    headers = [
        'F1 - Mean', 'F1 - SD', 'Precision - Mean', 'Precision - SD',
        'Recall - Mean', 'Recall - SD', 'Test Samples - Mean',
        'Test Samples - SD'
    ]

    for i, c in enumerate(classes):

        results[c] = [
            f1_scores_means[i], f1_scores_stds[i], precision_means[i],
            precision_stds[i], recalls_means[i], recalls_stds[i],
            supports_means[i], support_stds[i]
        ]

    return results, headers


def print_results(results, headers):
    table = []
    for k, v in results.items():
        line = prepend_string_to_array(k, v)
        table.append(line)
    headers.insert(0, 'Class')
    return tabulate(table, headers=headers)
