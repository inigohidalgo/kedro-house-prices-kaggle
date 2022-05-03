from sklearn import metrics


def get_scoring_function(function_name):
    metrics_dict = {
        "r2": metrics.r2_score,
        "mse": metrics.mean_squared_error,
        "accuracy": metrics.accuracy_score,
    }
    return metrics_dict[function_name]
