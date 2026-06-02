# src/training/test.py

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from models.efficientNet import EfficientNetSkinClassifier
from src.training.loader import build_datasets
from src.training.losses import get_loss
from src.training.metrics import (
    create_val_metrics,
    reset_metrics,
    get_metric_results,
)
from src.training.validate import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def setup_gpu():
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
    y_true_labels = prediction_data["y_true_labels"]
    y_pred_labels = prediction_data["y_pred_labels"]
    confidences = prediction_data["confidences"]
    probabilities = prediction_data["probabilities"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
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


def save_confusion_matrix_image(cm, class_names, output_path, normalize=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_to_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.nan_to_num(cm_to_plot)
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        cm_to_plot = cm
        title = "Confusion Matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=(18, 16))

    im = ax.imshow(cm_to_plot, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = cm_to_plot[i, j]

            if normalize:
                text_value = format(value, fmt)
            else:
                text_value = format(int(value), fmt)

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_class_f1_bar_image(report, class_names, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f1_scores = []

    for class_name in class_names:
        f1_scores.append(report[class_name]["f1-score"])

    sorted_items = sorted(
        zip(class_names, f1_scores),
        key=lambda x: x[1],
    )

    sorted_class_names = [item[0] for item in sorted_items]
    sorted_f1_scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(sorted_class_names, sorted_f1_scores)

    ax.set_xlabel("F1-score")
    ax.set_ylabel("Class")
    ax.set_title("F1-score per Class")
    ax.set_xlim(0, 1)

    for i, score in enumerate(sorted_f1_scores):
        ax.text(
            score + 0.01,
            i,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def test_worker():
    setup_gpu()

    output_dir = OUTPUT_DIR

    config_path = output_dir / "config.json"
    class_names_path = output_dir / "class_names.json"
    best_model_path = output_dir / "final_model.weights.h5"

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {output_dir}")
    print(f"Config path  : {config_path}")
    print(f"Class path   : {class_names_path}")
    print(f"Model path   : {best_model_path}")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file tidak ditemukan: {config_path}. "
            "Pastikan training.py sudah dijalankan."
        )

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file tidak ditemukan: {class_names_path}. "
            "Pastikan training.py sudah menyimpan class_names.json."
        )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model tidak ditemukan: {best_model_path}. "
            "Pastikan training.py sudah menghasilkan final_model.weights.h5."
        )

    config = load_json(config_path)
    class_names = load_json(class_names_path)
    num_classes = len(class_names)

    print("Loading test dataset...")

    datasets = build_datasets(config)
    test_ds = datasets[2]

    print("Loading model weights...")

    model = EfficientNetSkinClassifier(
        num_classes=num_classes,
        backbone=config.get("model_name", "efficientnet_b0"),
        image_channels=config.get("image_channels", 3),
        input_size=config.get("image_size", 224),
        pretrained=False,
        dropout=config.get("dropout", 0.3),
        freeze_backbone=config.get("freeze_backbone", False),
        hidden_dim=config.get("hidden_dim", 512),
    )

    dummy_input = tf.zeros(
        (
            1,
            config.get("image_size", 224),
            config.get("image_size", 224),
            config.get("image_channels", 3),
        ),
        dtype=tf.float32,
    )

    _ = model(dummy_input, training=False)

    model.load_weights(best_model_path)

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
        labels=list(range(num_classes)),
    )

    test_results_path = output_dir / "test_metrics.json"
    report_json_path = output_dir / "classification_report.json"
    report_txt_path = output_dir / "classification_report.txt"
    predictions_csv_path = output_dir / "test_predictions.csv"
    confusion_matrix_png_path = output_dir / "confusion_matrix.png"
    confusion_matrix_norm_png_path = output_dir / "confusion_matrix_normalized.png"
    class_f1_png_path = output_dir / "class_f1_score.png"

    save_json(test_results, test_results_path)
    save_json(report, report_json_path)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    save_predictions_csv(
        prediction_data=prediction_data,
        class_names=class_names,
        output_path=predictions_csv_path,
    )

    save_confusion_matrix_image(
        cm=cm,
        class_names=class_names,
        output_path=confusion_matrix_png_path,
        normalize=False,
    )

    save_confusion_matrix_image(
        cm=cm,
        class_names=class_names,
        output_path=confusion_matrix_norm_png_path,
        normalize=True,
    )

    save_class_f1_bar_image(
        report=report,
        class_names=class_names,
        output_path=class_f1_png_path,
    )

    print("\nClassification Report:")
    print(report_text)

    print("\nFiles saved:")
    print(f"- {test_results_path}")
    print(f"- {report_json_path}")
    print(f"- {report_txt_path}")
    print(f"- {predictions_csv_path}")
    print(f"- {confusion_matrix_png_path}")
    print(f"- {confusion_matrix_norm_png_path}")
    print(f"- {class_f1_png_path}")

    print("\nTarget Check:")

    accuracy_pass = test_results["accuracy"] >= 0.85
    mae_pass = test_results["mae"] <= 0.02

    print(f"Accuracy >= 85% : {'PASS' if accuracy_pass else 'NOT PASS'}")
    print(f"MAE <= 0.02     : {'PASS' if mae_pass else 'NOT PASS'}")


if __name__ == "__main__":
    test_worker()