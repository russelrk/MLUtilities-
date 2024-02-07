import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from sklearn.utils.validation import check_is_fitted
from typing import Dict, List, Optional, Union

def clustering_reports(
    y_true: np.ndarray, 
    y_preds: np.ndarray, 
    data: np.ndarray,
    metrics: Optional[List[str]] = None,
    distance_metric: str = 'euclidean',
    include_sparse: bool = False
) -> Dict[str, Union[float, None]]:
    """
    Computes various clustering metrics including ARI, NMI, Silhouette Score, 
    Calinski Harabasz Score, and optionally Davies-Bouldin Index.

    Args:
        y_true (np.ndarray): The ground truth cluster labels.
        y_preds (np.ndarray): The predicted cluster labels.
        data (np.ndarray): The dataset (used for silhouette, Calinski Harabasz, and Davies-Bouldin scores).
        metrics (Optional[List[str]]): List of metrics to compute. If None, computes all supported metrics.
        distance_metric (str): The distance metric to use for the Silhouette Score. Defaults to 'euclidean'.
        include_sparse (bool): If True, allows the function to support sparse data input. Defaults to False.

    Returns:
        Dict[str, Union[float, None]]: A dictionary containing the requested metrics, with None for any metric that fails.
    """
    supported_metrics = {
        "Adjusted Rand Index": adjusted_rand_score,
        "Normalized Mutual Information": normalized_mutual_info_score,
        "Silhouette Score": silhouette_score,
        "Calinski Harabasz Score": calinski_harabasz_score,
        "Davies-Bouldin Score": davies_bouldin_score
    }
    
    # Default to all metrics if none specified
    if metrics is None:
        metrics = list(supported_metrics.keys())
    
    results = {}
    for metric_name in metrics:
        try:
            if metric_name in supported_metrics:
                if metric_name in ["Silhouette Score", "Calinski Harabasz Score", "Davies-Bouldin Score"]:
                    # Check for sparse data support
                    if include_sparse:
                        data_input = data
                    else:
                        data_input = data.toarray() if sp.issparse(data) else data
                    if metric_name == "Silhouette Score":
                        score = supported_metrics[metric_name](data_input, y_preds, metric=distance_metric)
                    else:
                        score = supported_metrics[metric_name](data_input, y_preds)
                else:
                    score = supported_metrics[metric_name](y_true, y_preds)
                results[metric_name] = score
            else:
                results[metric_name] = None
                print(f"Metric {metric_name} is not supported.")
        except Exception as e:
            print(f"An error occurred while computing {metric_name}: {e}")
            results[metric_name] = None
            
    return results
