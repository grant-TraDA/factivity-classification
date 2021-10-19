import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def summarize_model_per_factive(y_pred, y_true, factive_ind):
    result = pd.DataFrame({
        "y_pred": y_pred,
        "y_true": y_true,
        "factive_ind": factive_ind 
    })
    result["correct"] = (y_pred == y_true)
    result.group_by(["y_true", "factive_ind"]).agg(
        n=("y_true", "count"),
        sum_correct=("correct", "sum"),
        acc=("correct", "mean"),
    )
    return result


def plot_feature_importance(features, importances, save_path, top_n=None):
    indices = np.argsort(importances)
    if top_n:
        indices = indices[-top_n:]
    plt.title('Impurity-based Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='#1e4585', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance')
    plt.ylabel(None)
    plt.show()
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
