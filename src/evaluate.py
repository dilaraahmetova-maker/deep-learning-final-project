from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate main classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in readable format.
    """
    print(f"{model_name} Results")
    print("-" * 40)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1-score:", round(metrics["f1"], 4))


def print_classification_report(y_true, y_pred):
    """
    Print sklearn classification report.
    """
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["negative", "positive"],
            zero_division=0
        )
    )


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"]
    )

    disp.plot(values_format="d")
    plt.title(title)
    plt.show()


def save_results(metrics, model_name, output_path):
    """
    Save model metrics into a CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    results_df = pd.DataFrame([
        {
            "model": model_name,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"]
        }
    ])

    results_df.to_csv(output_path, index=False)

    print("Results saved to:")
    print(output_path)

    return results_df
