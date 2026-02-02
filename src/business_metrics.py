import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix

def find_optimal_threshold(y_true, y_proba, 
                           retention_cost=100, 
                           acquisition_cost=500):
    """
    retention_cost: cost to keep a customer (offer, discount, outreach)
    acquisition_cost: cost to replace a churned customer
    """

    thresholds = np.linspace(0.1, 0.9, 50)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Business cost model
        cost = (fp * retention_cost) + (fn * acquisition_cost)

        results.append({
            "threshold": t,
            "fp": fp,
            "fn": fn,
            "business_cost": cost
        })

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[results_df["business_cost"].idxmin()]

    print("\n=== BUSINESS-OPTIMAL THRESHOLD ===")
    print(f"Best Threshold: {best_row['threshold']:.3f}")
    print(f"False Positives (unnecessary retention): {int(best_row['fp'])}")
    print(f"False Negatives (missed churn): {int(best_row['fn'])}")
    print(f"Estimated Business Cost: ${int(best_row['business_cost'])}")

    return best_row, results_df
