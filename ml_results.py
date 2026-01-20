def get_model_metrics(model):
    """
    Returns evaluation metrics for the selected ML model.
    These values come from offline training done in Colab.
    """

    if model == "Random Forest":
        return {
            "Accuracy": 0.68,
            "AUC": 0.73,
            "Precision": 0.67,
            "Recall": 0.72,
            "F1": 0.69
        }

    elif model == "XGBoost":
        return {
            "Accuracy": 0.67,
            "AUC": 0.73,
            "Precision": 0.67,
            "Recall": 0.66,
            "F1": 0.66
        }

    else:
        return {
            "Accuracy": "N/A",
            "AUC": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1": "N/A"
        }


