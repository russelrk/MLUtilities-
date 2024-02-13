import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Union, Optional
from tabulate import tabulate

def determine_optimal_threshold(use_youden_index: bool, target_value: Optional[float], 
                                target_sensitivity: bool, true_positive_rate: np.ndarray, 
                                false_positive_rate: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Determine the optimal threshold value based on the specified criterion.
    """
    if use_youden_index:
        return thresholds[np.argmax(true_positive_rate - false_positive_rate)]
    elif target_value is not None:
        if target_sensitivity:
            index = np.abs(true_positive_rate - target_value).argmin()
        else:
            index = np.abs(false_positive_rate - (1 - target_value)).argmin()
        return thresholds[index]
    else:
        return 0.5

def compute_classification_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                                   use_youden_index: Optional[bool] = False,
                                   target_value: Optional[float] = None,
                                   target_sensitivity: Optional[bool] = False,
                                   positive_label: Optional[int] = 1,
                                   display_confusion_matrix: Optional[bool] = False,
                                   calculate_weighted_f1: Optional[bool] = False) -> Dict[str, Union[float, str]]:
    """
    Computes and returns various classification metrics.
    """
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("Input arrays must have the same length.")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=positive_label)
    optimal_threshold = determine_optimal_threshold(use_youden_index, target_value, target_sensitivity, tpr, fpr, thresholds)
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    if display_confusion_matrix:
        print(tabulate([["Predicted Negative", tn, fp], ["Predicted Positive", fn, tp]], 
                       headers=["", "Actual Negative", "Actual Positive"], tablefmt='grid'))

    metrics = {
        "ROC-AUC Score": roc_auc_score(y_true, y_scores),
        "Accuracy Score": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if tn + fp > 0 else float('nan'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted') if calculate_weighted_f1 else f1_score(y_true, y_pred)
    }
    
    return metrics
