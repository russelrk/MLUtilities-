import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from typing import Dict, Union, Optional
from tabulate import tabulate

def determine_threshold(youden_index: bool, target_value: Optional[float], 
                        sen: bool, tpr: np.ndarray, fpr: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Determine threshold value based on the specified criterion.
    """
    if youden_index:
        return thresholds[np.argmax(tpr - fpr)]
    elif target_value is not None:
        return thresholds[(np.abs(tpr - target_value) if sen else np.abs(1 - fpr - target_value)).argmin()]
    else:
        return 0.5

def classification_results(
    y_true: np.ndarray, y_preds: np.ndarray, 
    youden_index: Optional[bool] = False,
    target_value: Optional[float] = None,
    sen: Optional[bool] = False,
    positive_label: Optional[int] = 1,
    display_cm: Optional[bool] = False,
    weighted_f1: Optional[bool] = False
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes various classification metrics including AUC-ROC, sensitivity, specificity, and accuracy.

    Args:
        y_true (np.ndarray): The ground truth binary labels.
        y_preds (np.ndarray): The predicted scores.
        youden_index (bool, optional): If true, uses Youden's Index to determine optimal threshold. 
        target_value (float, optional): Target specificity or sensitivity.
        sen (bool, optional): If true, target value refers to sensitivity, otherwise specificity.
        positive_label (int, optional): The label considered as positive in the dataset. Defaults to 1.
        display_cm (bool, optional): Whether to display confusion matrix.
        weighted_f1 (bool, optional): Whether to calcualted weighted F1 score

    Returns:
        Dict[str, Union[float, np.ndarray]]: A dictionary containing the ROC-AUC score, accuracy score, precision, sensitivity, and specificity.
    """
    try:
        # Check if inputs have the correct shape
        if y_true.shape[0] != y_preds.shape[0]:
            raise ValueError("Input arrays 'y_true' and 'y_preds' must have the same shape.")

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_preds)

        # Determine the optimal threshold
        OT = determine_threshold(youden_index, target_value, sen, tpr, fpr, thresholds)

        # Generate binary predictions using the threshold
        y_preds_binary = (y_preds > OT).astype(int)

        # Compute confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds_binary).ravel()

        # If needed, display the confusion matrix in a tabulated form
        if display_cm:
            labels = ["", "Actual Negative", "Actual Positive"]
            table_data = [["Predicted Negative", tn, fp],
                          ["Predicted Positive", fn, tp]] # Note the change from fp to fn
            print(tabulate(table_data, headers=labels, tablefmt='grid'))

        # Calculate the required metrics
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_true, y_preds_binary) if not weighted_f1 else f1_score(y_true, y_preds_binary, average = 'weighted')

        # Return the calculated metrics in dictionary format
        return {
            "ROC-AUC Score": roc_auc_score(y_true, y_preds),
            "Accuracy Score": accuracy_score(y_true, y_preds_binary),
            "Precision": precision,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "f1_score": f1
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


