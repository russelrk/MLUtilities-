import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union
import math

def regression_reports(
    y_true: np.ndarray, y_preds: np.ndarray
) -> Dict[str, float]:
    """
    Computes various regression metrics including MAE, MSE, RMSE, and R-squared.

    Args:
        y_true (np.ndarray): The ground truth labels.
        y_preds (np.ndarray): The predicted scores.

    Returns:
        Dict[str, float]: A dictionary containing the MAE, MSE, RMSE, and R-squared.
    """
    try:
        # Check if inputs have the correct shape
        if y_true.shape[0] != y_preds.shape[0]:
            raise ValueError("Input arrays 'y_true' and 'y_preds' must have the same shape.")

        # Compute Mean Absolute Error
        mae = mean_absolute_error(y_true, y_preds)
        
        # Compute Mean Squared Error
        mse = mean_squared_error(y_true, y_preds)
        
        # Compute Root Mean Squared Error
        rmse = math.sqrt(mse)
        
        # Compute R-squared
        r2 = r2_score(y_true, y_preds)

        # Return the calculated metrics in dictionary format
        return {
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "R-squared": r2
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
