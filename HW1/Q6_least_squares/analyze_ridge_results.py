import pandas as pd
import seaborn as sns


def analyze_results(ridge_obj, cutoff):

    truth = ridge_obj.y.toarray()[:,0]

    # categorize the info
    y_preds_for_3s = ridge_obj.y_preds[truth ==1]
    y_preds_for_other_numbers = ridge_obj.y_preds[truth == 0]

    true_positives = y_preds_for_3s [y_preds_for_3s >= cutoff]
    true_negatives = \
        y_preds_for_other_numbers[y_preds_for_other_numbers <= cutoff]

    false_positives = y_preds_for_3s [y_preds_for_3s < cutoff]
    false_negatives = \
        y_preds_for_other_numbers[y_preds_for_other_numbers > cutoff]

    # check that the results sum up correctly.
    assert(len(true_positives) + len(true_negatives) +
           len(false_positives) + len(false_negatives) == len(truth))

    call_counts = {"true + count": len(true_positives),
                   "true - count": len(true_negatives),
                   "false + count": len(false_positives),
                   "false - count": len(false_negatives)}
    y_vals = {"true +": true_positives, "true -": true_negatives,
            "false +": false_positives, "false -": false_negatives}

    # you get one point for every correct prediction and 0 for every wrong one.
    loss_01 = len(false_positives) + len(false_negatives)
    loss_01_norm = loss_01/len(truth)

    return {"call_counts": call_counts, "y_vals": y_vals,
            "loss_01": loss_01, "loss_01_norm": loss_01_norm}

def plot_results(preds, labels):

    to_compare = pd.DataFrame({"predictions": preds,
                               "labels": labels})

    print(to_compare.head())
    print(to_compare.shape)

    sns.violinplot(x='labels', y='predictions',
                   data=to_compare)
