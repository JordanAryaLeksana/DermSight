# src/training/validate.py

import tensorflow as tf
from tqdm import tqdm 
def validate_one_epoch(
    model,
    val_ds,
    loss_fn,
    val_metrics,
    epoch=None,
    total_epochs=None,
):
    """
    Custom evaluation loop untuk validation dengan progress bar.
    """

    total_steps = tf.data.experimental.cardinality(val_ds).numpy()

    progress_bar = tqdm(
        val_ds,
        total=total_steps if total_steps > 0 else None,
        desc=f"Validation Epoch {epoch}/{total_epochs}" if epoch else "Validation",
        unit="batch",
        ncols=120,
    )

    for images, labels in progress_bar:
        predictions = model(images, training=False)

        loss = loss_fn(labels, predictions)
        loss = tf.reduce_mean(loss)

        val_metrics["loss"].update_state(loss)
        val_metrics["accuracy"].update_state(labels, predictions)
        val_metrics["mae"].update_state(labels, predictions)

        if "macro_f1" in val_metrics:
            val_metrics["macro_f1"].update_state(labels, predictions)

        progress_bar.set_postfix({
            "loss": f"{val_metrics['loss'].result().numpy():.4f}",
            "acc": f"{val_metrics['accuracy'].result().numpy():.4f}",
            "mae": f"{val_metrics['mae'].result().numpy():.4f}",
            "f1": f"{val_metrics['macro_f1'].result().numpy():.4f}"
            if "macro_f1" in val_metrics else "N/A",
        })

def evaluate_dataset(
    model,
    dataset,
    loss_fn,
    metrics,
):

    for images, labels in dataset:
        predictions = model(images, training=False)

        loss = loss_fn(labels, predictions)
        loss = tf.reduce_mean(loss)

        metrics["loss"].update_state(loss)
        metrics["accuracy"].update_state(labels, predictions)
        metrics["mae"].update_state(labels, predictions)

        if "macro_f1" in metrics:
            metrics["macro_f1"].update_state(labels, predictions)