# src/training/validate.py

import tensorflow as tf


def validate_one_epoch(
    model,
    val_ds,
    loss_fn,
    val_metrics,
):

    for images, labels in val_ds:
        predictions = model(images, training=False)

        loss = loss_fn(labels, predictions)
        loss = tf.reduce_mean(loss)

        val_metrics["loss"].update_state(loss)
        val_metrics["accuracy"].update_state(labels, predictions)
        val_metrics["mae"].update_state(labels, predictions)

        if "macro_f1" in val_metrics:
            val_metrics["macro_f1"].update_state(labels, predictions)


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