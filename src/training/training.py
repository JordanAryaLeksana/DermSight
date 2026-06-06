
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

from tqdm import tqdm
from src.training.loader import build_datasets
from src.models.modelFactory import build_model
from src.training.losses import get_loss
from src.training.metrics import (
    create_train_metrics,
    create_val_metrics,
    reset_metrics,
    get_metric_results,
)

from src.training.validate import validate_one_epoch
from src.training.tensorboard_utils import (
    create_tensorboard_writer,
    write_epoch_logs,
)
from src.training.lr_scheduler import ManualReduceLROnPlateau
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=6144  # MB
                    )
                ],
            )

            logical_gpus = tf.config.list_logical_devices("GPU")
            print("Logical GPUs:", logical_gpus)

        except RuntimeError as e:
            print(e)


def train_one_epoch(
    model,
    train_ds,
    loss_fn,
    optimizer,
    train_metrics,
    epoch=None,
    total_epochs=None,
    class_weights=None,
):

    total_steps = tf.data.experimental.cardinality(train_ds).numpy()

    progress_bar = tqdm(
        train_ds,
        total=total_steps if total_steps > 0 else None,
        desc=f"Training Epoch {epoch}/{total_epochs}" if epoch else "Training",
        unit="batch",
        ncols=120,
    )

    for step, (images, labels) in enumerate(progress_bar):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)

            loss_per_sample = loss_fn(labels, predictions)

            if class_weights is not None:
                sample_weights = tf.reduce_sum(
                    tf.cast(labels, tf.float32) * class_weights,
                    axis=-1
                )
                loss_per_sample = loss_per_sample * sample_weights

            loss = tf.reduce_mean(loss_per_sample)

        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_metrics["loss"].update_state(loss)
        train_metrics["accuracy"].update_state(labels, predictions)
        train_metrics["mae"].update_state(labels, predictions)

        if "macro_f1" in train_metrics:
            train_metrics["macro_f1"].update_state(labels, predictions)

        progress_bar.set_postfix(
            {
                "loss": f"{train_metrics['loss'].result().numpy():.4f}",
                "acc": f"{train_metrics['accuracy'].result().numpy():.4f}",
                "mae": f"{train_metrics['mae'].result().numpy():.4f}",
                "f1": (
                    f"{train_metrics['macro_f1'].result().numpy():.4f}"
                    if "macro_f1" in train_metrics
                    else "N/A"
                ),
            }
        )


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)




def train_worker():
    config = {
        "train_dir": "src/data/train",
        "val_dir": "src/data/val",
        "test_dir": "src/data/test",
        # Dataset/image config
        "image_size": 224,
        "image_channels": 3,
        "input_size": (224, 224),
        "input_shape": (224, 224, 3),
        "batch_size": 16,
        # Model config
        "model_name": "efficientnet_b2",
        "pretrained": True,
        "weights": "imagenet",  # use None kalau tidak mau pretrained
        "freeze_backbone": False,
        "trainable": False, 
        "dropout": 0.2,
        "hidden_dim": 512,
        # Training config
        "epochs": 50,
        "loss_name": "cross_entropy",
        "learning_rate": 1e-5,
        "min_learning_rate": 1e-7,
        "lr_factor": 0.3,
        "lr_patience": 3,
        "early_stopping_patience": 10,
        # Output/logging
        "output_dir": "src/outputs",
        "log_dir": "src/logs",
        # Runtime
        "seed": 42,
        # "use_mixed_precision": False,
    }
    tf.random.set_seed(seed=config["seed"])

    # setup_gpu(use_mixed_precision=config["use_mixed_precision"])

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    print("\n loading datasets")

    train_ds, val_ds, test_ds, class_names, num_classes, class_weight, class_counts = build_datasets(config)
    
    # class_weights = compute_class_weights_from_directory(
    #     config["train_dir"],
    #     class_names,
    # )
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")

    print("\nBuilding model...")

    model = build_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        image_channels=config["image_channels"],
        input_size=config["image_size"],
        pretrained=config["pretrained"],
        dropout=config["dropout"],
        freeze_backbone=config["freeze_backbone"],
        hidden_dim=config["hidden_dim"],
    )

    loss_fn = get_loss(config["loss_name"])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    lr_scheduler = ManualReduceLROnPlateau(
        optimizer=optimizer,
        monitor="val_loss",
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        min_lr=config["min_learning_rate"],
        min_delta=1e-4,
        verbose=True,
    )

    train_metrics = create_train_metrics(num_classes)
    val_metrics = create_val_metrics(num_classes)

    writer, tensorboard_log_dir = create_tensorboard_writer(log_dir=config["log_dir"])

    print(f"TensorBoard log dir: {tensorboard_log_dir}")

    best_val_macro_f1 = 0.0
    counter = 0

    best_model_path = os.path.join(config["output_dir"], "best_model.weights.h5")

    final_model_path = os.path.join(config["output_dir"], "final_model.weights.h5")

    history = []

    save_json(class_names, os.path.join(config["output_dir"], "class_names.json"))

    save_json(config, os.path.join(config["output_dir"], "config.json"))

    print("\nStart training...")

    for epoch in range(1, config["epochs"] + 1):
        reset_metrics(train_metrics)
        reset_metrics(val_metrics)

        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_one_epoch(
            model=model,
            train_ds=train_ds,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_metrics=train_metrics,
            epoch=epoch,
            total_epochs=config["epochs"],
            class_weights=class_weight,
        )

        validate_one_epoch(
            model=model,
            val_ds=val_ds,
            loss_fn=loss_fn,
            val_metrics=val_metrics,
            epoch=epoch,
            total_epochs=config["epochs"],
        )

        train_results = get_metric_results(train_metrics)
        val_results = get_metric_results(val_metrics)

        lr_scheduler.step(val_results["loss"])

        learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))

        write_epoch_logs(
            writer=writer,
            epoch=epoch,
            train_results=train_results,
            val_results=val_results,
            learning_rate=learning_rate,
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "train_mae": train_results["mae"],
            "train_macro_f1": train_results.get("macro_f1", None),
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_mae": val_results["mae"],
            "val_macro_f1": val_results.get("macro_f1", None),
            "learning_rate": learning_rate,
        }

        history.append(epoch_log)
        print(
            f"train_loss: {train_results['loss']:.4f} | "
            f"train_acc: {train_results['accuracy']:.4f} | "
            f"train_mae: {train_results['mae']:.4f} | "
            f"train_f1: {train_results.get('macro_f1', 0.0):.4f} | "
            f"val_loss: {val_results['loss']:.4f} | "
            f"val_acc: {val_results['accuracy']:.4f} | "
            f"val_mae: {val_results['mae']:.4f} | "
            f"val_f1: {val_results.get('macro_f1', 0.0):.4f}"
        )

        current_val_macro_f1 = val_results.get("macro_f1", 0.0)

        if current_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_val_macro_f1
            counter = 0

            model.save_weights(best_model_path)
            print(
                f"Best model saved to {best_model_path} "
                f"with val_macro_f1: {best_val_macro_f1:.4f}"
            )
        else:
            counter += 1
            print(
                f"No improvement. "
                f"Patience: {counter}/{config['early_stopping_patience']}"
            )
        save_json(history, os.path.join(config["output_dir"], "training_history.json"))

        if counter >= config["early_stopping_patience"]:
            print("\nEarly stopping triggered.")
            break

    model.save_weights(final_model_path)

    print("\nTraining finished.")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Best model path: {best_model_path}")
    print(f"Final model path: {final_model_path}")
    print(f"TensorBoard logs: {tensorboard_log_dir}")


if __name__ == "__main__":
    train_worker()
