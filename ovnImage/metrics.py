import numpy as np

import sklearn.metrics as metrics


def union_bounding_box(a: tuple, b: tuple) -> tuple:
    """
    Get the bounding box resulting of the union of two bounding boxes.

    :param a: Tuple defining a bounding box using the format (x, y, width, height)
    :param b: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return x, y, w, h


def intersection_bounding_box(a, b) -> tuple:
    """
    Get the bounding box resulting of the intersection of two bounding boxes.
    If no intersection return a tuple with (0, 0, 0, 0).

    :param a: Tuple defining a bounding box using the format (x, y, width, height)
    :param b: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return 0, 0, 0, 0
    else:
        return x, y, w, h


def IoU_bounding_box(bb1, bb2) -> float:
    """
    Get the intersection over union (IoU) metric value of two given bounding boxes.

    :param bb1: Tuple defining a bounding box using the format (x, y, width, height)
    :param bb2: Tuple defining a bounding box using the format (x, y, width, height)
    :return:
    """
    x, y, w, h = union_bounding_box(bb1, bb2)
    area_union = w * h

    x, y, w, h = intersection_bounding_box(bb1, bb2)
    area_intersection = w*h

    if area_union == 0:
        return 0
    else:
        return area_intersection / area_union


def get_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Get a dictionary with some of the main classification metrics from an array
    of predicted labels and its true labels.

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels
    :return:
    """
    J = metrics.jaccard_score(y_true, y_pred)

    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    prec_avg, rec_avg, f1_avg, _ = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                           average="weighted")
    return {
        'Jaccard': J,
        'Precision other': precision[0],
        'Recall other': recall[0],
        'F1 other': f1[0],
        'Support other': support[0],
        'Precision target': precision[1],
        'Recall target': recall[1],
        'F1 target': f1[1],
        'Support target': support[1],
        'Precision Weighted average': prec_avg,
        'Recall Weighted average': rec_avg,
        'F1 Weighted average': f1_avg
    }


def print_classification_stats(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> dict:
    """
    Print in stdout some of the main classification metrics from an array of predicted labels and its true labels.
    Get a dictionary with the computed classification metrics.

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels
    :param labels: list of strings
        Optional display names matching the labels (same order).
    :return:
    """
    print(metrics.classification_report(y_true, y_pred, target_names=labels))
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted', pos_label=True)

    avg_scores = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                         average="weighted")
    print(avg_scores)

    print("Confussion matrix: ")
    print(metrics.confusion_matrix(y_true, y_pred))

    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    print("Cohen kappa score: " + str(cohen_kappa))
    return {
        "F1": f1,
        "Recall(TPR)": recall,
        "Precision": precision,
        "Precision_avg": str(avg_scores[0]),
        "Recall_avg": str(avg_scores[1]),
        "F1_avg": str(avg_scores[2]),
        "Support_total": str(avg_scores[3]),
        "cohen_kappa": cohen_kappa
    }
