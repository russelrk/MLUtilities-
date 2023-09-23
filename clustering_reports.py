import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from typing import Dict, Union

def clustering_reports(
    y_true: np.ndarray, y_preds: np.ndarray, data: np.ndarray
) -> Dict[str, float]:
    """
    Computes various clustering metrics including ARI, NMI, Silhouette Score, and Calinski Harabasz Score.

    Args:
        y_true (np.ndarray): The ground truth cluster labels.
        y_preds (np.ndarray): The predicted cluster labels.
        data (np.ndarray): The dataset (used for silhouette and Calinski Harabasz scores).

    Returns:
        Dict[str, float]: A dictionary containing the ARI, NMI, Silhouette Score, and Calinski Harabasz Score.
    """
    try:
        # Check if inputs have the correct shape
        if y_true.shape[0] != y_preds.shape[0] or y_true.shape[0] != data.shape[0]:
            raise ValueError("Input arrays 'y_true', 'y_preds', and 'data' must have the same number of samples.")

        # Compute Adjusted Rand Index
        ari = adjusted_rand_score(y_true, y_preds)
        
        # Compute Normalized Mutual Information
        nmi = normalized_mutual_info_score(y_true, y_preds)
        
        # Compute Silhouette Score
        silhouette = silhouette_score(data, y_preds)
        
        # Compute Calinski Harabasz Score
        calinski_harabasz = calinski_harabasz_score(data, y_preds)

        # Return the calculated metrics in dictionary format
        return {
            "Adjusted Rand Index": ari,
            "Normalized Mutual Information": nmi,
            "Silhouette Score": silhouette,
            "Calinski Harabasz Score": calinski_harabasz
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
