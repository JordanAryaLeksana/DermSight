# src/training/test.py

import os
import json
import csv
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.training.loader import build_datasets
from src.training.losses import get_loss
from src.training.metrics import (
    create_val_metrics,
    reset_metrics,
    get_metric_results,
)
from src.training.validate import evaluate_dataset


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def setup_gpu():
    """
    Setup GPU agar TensorFlow tidak langsung mengambil semua VRAM.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(f"GPU detected: {gpus}")

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Could not set memory growth: {e}")
    else:
        print("No GPU detected. Evaluation will use CPU.")


def collect_predictions(model, test_ds, class_names):
    """
    Mengumpulkan y_true, y_pred, confidence, dan probability dari test set.
    """

    y_true_indices = []
    y_pred_indices = []
    confidences = []
    probabilities_all = []

    for images, labels in test_ds:
        predictions = model(images, training=False)

        probs = predictions.numpy()
        true_indices = np.argmax(labels.numpy(), axis=1)
        pred_indices = np.argmax(probs, axis=1)
        batch_confidences = np.max(probs, axis=1)

        y_true_indices.extend(true_indices.tolist())
        y_pred_indices.extend(pred_indices.tolist())
        confidences.extend(batch_confidences.tolist())
        probabilities_all.extend(probs.tolist())

    y_true_labels = [class_names[idx] for idx in y_true_indices]
    y_pred_labels = [class_names[idx] for idx in y_pred_indices]

    return {
        "y_true_indices": y_true_indices,
        "y_pred_indices": y_pred_indices,
        "y_true_labels": y_true_labels,
        "y_pred_labels": y_pred_labels,
        "confidences": confidences,
        "probabilities": probabilities_all,
    }


def save_predictions_csv(prediction_data, class_names, output_path):
    """
    Menyimpan hasil prediksi ke CSV.
    """

    y_true_labels = prediction_data["y_true_labels"]
    y_pred_labels = prediction_data["y_pred_labels"]
    confidences = prediction_data["confidences"]
    probabilities = prediction_data["probabilities"]

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "index",
            "true_label",
            "predicted_label",
            "confidence",
        ]

        for class_name in class_names:
            header.append(f"prob_{class_name}")

        writer.writerow(header)

        for idx, (true_label, pred_label, confidence, probs) in enumerate(
            zip(y_true_labels, y_pred_labels, confidences, probabilities)
        ):
            row = [
                idx,
                true_label,
                pred_label,
                confidence,
            ]

            row.extend(probs)
            writer.writerow(row)


def save_confusion_matrix_csv(cm, class_names, output_path):
    """
    Menyimpan confusion matrix dalam bentuk CSV.
    """

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["true/pred"] + class_names)

        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())


def test_worker():
    setup_gpu()

    output_dir = "outputs"

    config_path = os.path.join(output_dir, "config.json")
    class_names_path = os.path.join(output_dir, "class_names.json")
    best_model_path = os.path.join(output_dir, "best_model.keras")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file tidak ditemukan: {config_path}. "
            "Pastikan training.py sudah dijalankan."
        )

    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Class names file tidak ditemukan: {class_names_path}. "
            "Pastikan training.py sudah menyimpan class_names.json."
        )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Best model tidak ditemukan: {best_model_path}. "
            "Pastikan training.py sudah menghasilkan best_model.keras."
        )

    config = load_json(config_path)
    class_names = load_json(class_names_path)
    num_classes = len(class_names)

    print("Loading test dataset...")

    _, _, test_ds, _, _ = build_datasets(config)

    print("Loading best model...")

    model = tf.keras.models.load_model(
        best_model_path,
        compile=False
    )

    loss_fn = get_loss(config.get("loss_name", "focal_loss"))

    test_metrics = create_val_metrics(num_classes)
    reset_metrics(test_metrics)

    print("Evaluating test set...")

    evaluate_dataset(
        model=model,
        dataset=test_ds,
        loss_fn=loss_fn,
        metrics=test_metrics,
    )

    test_results = get_metric_results(test_metrics)

    print("\nTest Metrics:")
    print(f"Test Loss     : {test_results['loss']:.4f}")
    print(f"Test Accuracy : {test_results['accuracy']:.4f}")
    print(f"Test MAE      : {test_results['mae']:.4f}")

    if "macro_f1" in test_results:
        print(f"Test Macro F1 : {test_results['macro_f1']:.4f}")

    print("\nCollecting predictions...")

    prediction_data = collect_predictions(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
    )

    y_true = prediction_data["y_true_indices"]
    y_pred = prediction_data["y_pred_indices"]

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    test_results_path = os.path.join(output_dir, "test_metrics.json")
    report_json_path = os.path.join(output_dir, "classification_report.json")
    report_txt_path = os.path.join(output_dir, "classification_report.txt")
    predictions_csv_path = os.path.join(output_dir, "test_predictions.csv")
    confusion_matrix_csv_path = os.path.join(output_dir, "confusion_matrix.csv")

    save_json(test_results, test_results_path)
    save_json(report, report_json_path)

    with open(report_txt_path, "w") as f:
        f.write(report_text)

    save_predictions_csv(
        prediction_data=prediction_data,
        class_names=class_names,
        output_path=predictions_csv_path,
    )

    save_confusion_matrix_csv(
        cm=cm,
        class_names=class_names,
        output_path=confusion_matrix_csv_path,
    )

    print("\nClassification Report:")
    print(report_text)

    print("\nFiles saved:")
    print(f"- {test_results_path}")
    print(f"- {report_json_path}")
    print(f"- {report_txt_path}")
    print(f"- {predictions_csv_path}")
    print(f"- {confusion_matrix_csv_path}")

    print("\nTarget Check:")

    accuracy_pass = test_results["accuracy"] >= 0.85
    mae_pass = test_results["mae"] <= 0.02

    print(f"Accuracy >= 85% : {'PASS' if accuracy_pass else 'NOT PASS'}")
    print(f"MAE <= 0.02     : {'PASS' if mae_pass else 'NOT PASS'}")


if __name__ == "__main__":
    test_worker()