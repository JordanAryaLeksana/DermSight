
# import json
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

# from tqdm import tqdm
# from src.training.loader import build_datasets
# from src.models.modelFactory import build_model
# from src.training.losses import get_loss
# from src.training.metrics import (
#     create_train_metrics,
#     create_val_metrics,
#     reset_metrics,
#     get_metric_results,
# )

# from src.training.validate import validate_one_epoch
# from src.training.tensorboard_utils import (
#     create_tensorboard_writer,
#     write_epoch_logs,
# )
# from src.training.lr_scheduler import ManualReduceLROnPlateau
# tf.config.threading.set_inter_op_parallelism_threads(4)
# tf.config.threading.set_intra_op_parallelism_threads(4)

# def setup_gpu():
#     gpus = tf.config.list_physical_devices("GPU")
#     if gpus:
#         try:
#             tf.config.set_logical_device_configuration(
#                 gpus[0],
#                 [
#                     tf.config.LogicalDeviceConfiguration(
#                         memory_limit=6144  # MB
#                     )
#                 ],
#             )

#             logical_gpus = tf.config.list_logical_devices("GPU")
#             print("Logical GPUs:", logical_gpus)

#         except RuntimeError as e:
#             print(e)


# def train_one_epoch(
#     model,
#     train_ds,
#     loss_fn,
#     optimizer,
#     train_metrics,
#     epoch=None,
#     total_epochs=None,
#     class_weights=None,
# ):

#     total_steps = tf.data.experimental.cardinality(train_ds).numpy()

#     progress_bar = tqdm(
#         train_ds,
#         total=total_steps if total_steps > 0 else None,
#         desc=f"Training Epoch {epoch}/{total_epochs}" if epoch else "Training",
#         unit="batch",
#         ncols=120,
#     )

#     for step, (images, labels) in enumerate(progress_bar):
#         with tf.GradientTape() as tape:
#             predictions = model(images, training=True)

#             loss_per_sample = loss_fn(labels, predictions)

#             if class_weights is not None:
#                 sample_weights = tf.reduce_sum(
#                     tf.cast(labels, tf.float32) * class_weights,
#                     axis=-1
#                 )
#                 loss_per_sample = loss_per_sample * sample_weights

#             loss = tf.reduce_mean(loss_per_sample)

#         gradients = tape.gradient(loss, model.trainable_variables)

#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         train_metrics["loss"].update_state(loss)
#         train_metrics["accuracy"].update_state(labels, predictions)
#         train_metrics["mae"].update_state(labels, predictions)

#         if "macro_f1" in train_metrics:
#             train_metrics["macro_f1"].update_state(labels, predictions)

#         progress_bar.set_postfix(
#             {
#                 "loss": f"{train_metrics['loss'].result().numpy():.4f}",
#                 "acc": f"{train_metrics['accuracy'].result().numpy():.4f}",
#                 "mae": f"{train_metrics['mae'].result().numpy():.4f}",
#                 "f1": (
#                     f"{train_metrics['macro_f1'].result().numpy():.4f}"
#                     if "macro_f1" in train_metrics
#                     else "N/A"
#                 ),
#             }
#         )


# def save_json(data, path):
#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)




# def train_worker():
#     config = {
#         "train_dir": "src/data/train",
#         "val_dir": "src/data/val",
#         "test_dir": "src/data/test",
#         # Dataset/image config
#         "image_size": 224,
#         "image_channels": 3,
#         "input_size": (224, 224),
#         "input_shape": (224, 224, 3),
#         "batch_size": 16,
#         # Model config
#         "model_name": "efficientnet_b2",
#         "pretrained": True,
#         "weights": "imagenet",  # use None kalau tidak mau pretrained
#         "freeze_backbone": False,
#         "trainable": False, 
#         "dropout": 0.2,
#         "hidden_dim": 512,
#         # Training config
#         "epochs": 50,
#         "loss_name": "cross_entropy",
#         "learning_rate": 1e-5,
#         "min_learning_rate": 1e-7,
#         "lr_factor": 0.3,
#         "lr_patience": 3,
#         "early_stopping_patience": 10,
#         # Output/logging
#         "output_dir": "src/outputs",
#         "log_dir": "src/logs",
#         # Runtime
#         "seed": 42,
#         # "use_mixed_precision": False,
#     }
#     tf.random.set_seed(seed=config["seed"])

#     # setup_gpu(use_mixed_precision=config["use_mixed_precision"])

#     os.makedirs(config["output_dir"], exist_ok=True)
#     os.makedirs(config["log_dir"], exist_ok=True)

#     print("\n loading datasets")

#     train_ds, val_ds, test_ds, class_names, num_classes, class_weight, class_counts = build_datasets(config)
    
#     # class_weights = compute_class_weights_from_directory(
#     #     config["train_dir"],
#     #     class_names,
#     # )
#     print(f"Class names: {class_names}")
#     print(f"Number of classes: {num_classes}")

#     print("\nBuilding model...")

#     model = build_model(
#         model_name=config["model_name"],
#         num_classes=num_classes,
#         image_channels=config["image_channels"],
#         input_size=config["image_size"],
#         pretrained=config["pretrained"],
#         dropout=config["dropout"],
#         freeze_backbone=config["freeze_backbone"],
#         hidden_dim=config["hidden_dim"],
#     )

#     loss_fn = get_loss(config["loss_name"])
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

#     lr_scheduler = ManualReduceLROnPlateau(
#         optimizer=optimizer,
#         monitor="val_loss",
#         mode="min",
#         factor=config["lr_factor"],
#         patience=config["lr_patience"],
#         min_lr=config["min_learning_rate"],
#         min_delta=1e-4,
#         verbose=True,
#     )

#     train_metrics = create_train_metrics(num_classes)
#     val_metrics = create_val_metrics(num_classes)

#     writer, tensorboard_log_dir = create_tensorboard_writer(log_dir=config["log_dir"])

#     print(f"TensorBoard log dir: {tensorboard_log_dir}")

#     best_val_macro_f1 = 0.0
#     counter = 0

#     best_model_path = os.path.join(config["output_dir"], "best_model.weights.h5")

#     final_model_path = os.path.join(config["output_dir"], "final_model.weights.h5")

#     history = []

#     save_json(class_names, os.path.join(config["output_dir"], "class_names.json"))

#     save_json(config, os.path.join(config["output_dir"], "config.json"))

#     print("\nStart training...")

#     for epoch in range(1, config["epochs"] + 1):
#         reset_metrics(train_metrics)
#         reset_metrics(val_metrics)

#         print(f"\nEpoch {epoch}/{config['epochs']}")

#         train_one_epoch(
#             model=model,
#             train_ds=train_ds,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             train_metrics=train_metrics,
#             epoch=epoch,
#             total_epochs=config["epochs"],
#             class_weights=class_weight,
#         )

#         validate_one_epoch(
#             model=model,
#             val_ds=val_ds,
#             loss_fn=loss_fn,
#             val_metrics=val_metrics,
#             epoch=epoch,
#             total_epochs=config["epochs"],
#         )

#         train_results = get_metric_results(train_metrics)
#         val_results = get_metric_results(val_metrics)

#         lr_scheduler.step(val_results["loss"])

#         learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))

#         write_epoch_logs(
#             writer=writer,
#             epoch=epoch,
#             train_results=train_results,
#             val_results=val_results,
#             learning_rate=learning_rate,
#         )
#         epoch_log = {
#             "epoch": epoch,
#             "train_loss": train_results["loss"],
#             "train_accuracy": train_results["accuracy"],
#             "train_mae": train_results["mae"],
#             "train_macro_f1": train_results.get("macro_f1", None),
#             "val_loss": val_results["loss"],
#             "val_accuracy": val_results["accuracy"],
#             "val_mae": val_results["mae"],
#             "val_macro_f1": val_results.get("macro_f1", None),
#             "learning_rate": learning_rate,
#         }

#         history.append(epoch_log)
#         print(
#             f"train_loss: {train_results['loss']:.4f} | "
#             f"train_acc: {train_results['accuracy']:.4f} | "
#             f"train_mae: {train_results['mae']:.4f} | "
#             f"train_f1: {train_results.get('macro_f1', 0.0):.4f} | "
#             f"val_loss: {val_results['loss']:.4f} | "
#             f"val_acc: {val_results['accuracy']:.4f} | "
#             f"val_mae: {val_results['mae']:.4f} | "
#             f"val_f1: {val_results.get('macro_f1', 0.0):.4f}"
#         )

#         current_val_macro_f1 = val_results.get("macro_f1", 0.0)

#         if current_val_macro_f1 > best_val_macro_f1:
#             best_val_macro_f1 = current_val_macro_f1
#             counter = 0

#             model.save_weights(best_model_path)
#             print(
#                 f"Best model saved to {best_model_path} "
#                 f"with val_macro_f1: {best_val_macro_f1:.4f}"
#             )
#         else:
#             counter += 1
#             print(
#                 f"No improvement. "
#                 f"Patience: {counter}/{config['early_stopping_patience']}"
#             )
#         save_json(history, os.path.join(config["output_dir"], "training_history.json"))

#         if counter >= config["early_stopping_patience"]:
#             print("\nEarly stopping triggered.")
#             break

#     model.save_weights(final_model_path)

#     print("\nTraining finished.")
#     print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
#     print(f"Best model path: {best_model_path}")
#     print(f"Final model path: {final_model_path}")
#     print(f"TensorBoard logs: {tensorboard_log_dir}")


# if __name__ == "__main__":
#     train_worker()


import json
import logging
import os

# Set before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tqdm import tqdm

from src.models.modelFactory import build_model
from src.training.loader import build_datasets
from src.training.losses import get_loss
from src.training.lr_scheduler import ManualReduceLROnPlateau
from src.training.metrics import (
    create_train_metrics,
    create_val_metrics,
    get_metric_results,
    reset_metrics,
)
from src.training.tensorboard_utils import (
    create_tensorboard_writer,
    write_epoch_logs,
)


logger = tf.get_logger()
logger.setLevel(logging.WARNING)


def setup_runtime(
    use_mixed_precision: bool = True,
    gpu_memory_limit_mb: int | None = None,
) -> None:
    """Configure GPU before model/dataset execution starts."""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("GPU not detected. Training will run on CPU.")
    else:
        print("Physical GPUs:", gpus)

        for gpu in gpus:
            try:
                if gpu_memory_limit_mb is None:
                    # Prefer memory growth on a desktop GPU so TensorFlow does
                    # not reserve all VRAM immediately.
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=gpu_memory_limit_mb
                            )
                        ],
                    )
            except RuntimeError as exc:
                # Happens if the device was already initialized.
                print(f"GPU configuration warning: {exc}")

        print("Logical GPUs:", tf.config.list_logical_devices("GPU"))

    if use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(
            "Mixed precision:",
            tf.keras.mixed_precision.global_policy(),
        )
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print(
            "Precision policy:",
            tf.keras.mixed_precision.global_policy(),
        )


def optimize_dataset(
    dataset: tf.data.Dataset,
    training: bool,
) -> tf.data.Dataset:
    """
    Add safe tf.data runtime optimizations.

    We intentionally do not call cache() because the current machine has
    already shown host-RAM pressure during training.
    """
    options = tf.data.Options()
    options.autotune.enabled = True

    # Training does not require deterministic ordering after the loader has
    # already shuffled the dataset. Validation/test remain deterministic.
    options.deterministic = not training

    dataset = dataset.with_options(options)

    # If the loader already prefetches, an additional prefetch stage is still
    # safe; TensorFlow can pipeline producer/consumer work.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def dataset_cardinality(dataset: tf.data.Dataset) -> int | None:
    value = int(tf.data.experimental.cardinality(dataset).numpy())

    # UNKNOWN_CARDINALITY = -2, INFINITE_CARDINALITY = -1.
    if value < 0:
        return None

    return value


def metric_snapshot(metrics: dict) -> dict[str, str]:
    """Synchronize metric values to CPU only when progress needs updating."""
    snapshot = {
        "loss": f"{metrics['loss'].result().numpy():.4f}",
        "acc": f"{metrics['accuracy'].result().numpy():.4f}",
        "mae": f"{metrics['mae'].result().numpy():.4f}",
    }

    if "macro_f1" in metrics:
        snapshot["f1"] = (
            f"{metrics['macro_f1'].result().numpy():.4f}"
        )

    return snapshot


def create_train_step(
    model: tf.keras.Model,
    loss_fn,
    optimizer,
    train_metrics: dict,
    class_weights=None,
    use_mixed_precision: bool = True,
    use_xla: bool = False,
):
    """
    Compile forward pass, loss, backward pass, optimizer step, and metric
    updates into a TensorFlow graph.

    XLA stays disabled by default because this project currently runs on an
    RTX 5060 / Blackwell GPU and the normal TF graph path is the safer choice.
    """
    if class_weights is not None:
        class_weights = tf.convert_to_tensor(
            class_weights,
            dtype=tf.float32,
        )

    has_macro_f1 = "macro_f1" in train_metrics

    @tf.function(
        reduce_retracing=True,
        jit_compile=use_xla,
    )
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            raw_predictions = model(
                images,
                training=True,
            )

            # Keep loss/metrics numerically stable when the model runs in fp16.
            predictions = tf.cast(
                raw_predictions,
                tf.float32,
            )

            loss_per_sample = loss_fn(
                labels,
                predictions,
            )

            loss_per_sample = tf.cast(
                loss_per_sample,
                tf.float32,
            )

            if class_weights is not None:
                sample_weights = tf.reduce_sum(
                    tf.cast(labels, tf.float32)
                    * class_weights,
                    axis=-1,
                )

                loss_per_sample = (
                    loss_per_sample
                    * sample_weights
                )

            loss = tf.reduce_mean(loss_per_sample)

            if use_mixed_precision:
                scaled_loss = optimizer.scale_loss(loss)
            else:
                scaled_loss = loss

        gradients = tape.gradient(
            scaled_loss,
            model.trainable_variables,
        )

        # Ignore disconnected variables defensively.
        gradients_and_variables = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                model.trainable_variables,
            )
            if gradient is not None
        ]

        optimizer.apply_gradients(
            gradients_and_variables
        )

        # Metrics stay inside the graph. We only call .numpy() occasionally
        # from the outer Python loop when updating tqdm.
        train_metrics["loss"].update_state(loss)
        train_metrics["accuracy"].update_state(
            labels,
            predictions,
        )
        train_metrics["mae"].update_state(
            labels,
            predictions,
        )

        if has_macro_f1:
            train_metrics["macro_f1"].update_state(
                labels,
                predictions,
            )

        return loss

    return train_step


def create_val_step(
    model: tf.keras.Model,
    loss_fn,
    val_metrics: dict,
    use_xla: bool = False,
):
    has_macro_f1 = "macro_f1" in val_metrics

    @tf.function(
        reduce_retracing=True,
        jit_compile=use_xla,
    )
    def val_step(images, labels):
        raw_predictions = model(
            images,
            training=False,
        )

        predictions = tf.cast(
            raw_predictions,
            tf.float32,
        )

        loss_per_sample = loss_fn(
            labels,
            predictions,
        )

        loss = tf.reduce_mean(
            tf.cast(loss_per_sample, tf.float32)
        )

        val_metrics["loss"].update_state(loss)
        val_metrics["accuracy"].update_state(
            labels,
            predictions,
        )
        val_metrics["mae"].update_state(
            labels,
            predictions,
        )

        if has_macro_f1:
            val_metrics["macro_f1"].update_state(
                labels,
                predictions,
            )

        return loss

    return val_step


def train_one_epoch(
    train_ds: tf.data.Dataset,
    train_step_fn,
    train_metrics: dict,
    epoch: int,
    total_epochs: int,
    log_every_n_steps: int = 25,
) -> None:
    total_steps = dataset_cardinality(train_ds)

    progress_bar = tqdm(
        train_ds,
        total=total_steps,
        desc=f"Training Epoch {epoch}/{total_epochs}",
        unit="batch",
        ncols=120,
    )

    for step, (images, labels) in enumerate(
        progress_bar,
        start=1,
    ):
        train_step_fn(images, labels)

        should_log = (
            step == 1
            or step % log_every_n_steps == 0
            or (
                total_steps is not None
                and step == total_steps
            )
        )

        if should_log:
            progress_bar.set_postfix(
                metric_snapshot(train_metrics),
                refresh=False,
            )


def validate_one_epoch_fast(
    val_ds: tf.data.Dataset,
    val_step_fn,
    val_metrics: dict,
    epoch: int,
    total_epochs: int,
    log_every_n_steps: int = 50,
) -> None:
    total_steps = dataset_cardinality(val_ds)

    progress_bar = tqdm(
        val_ds,
        total=total_steps,
        desc=f"Validation Epoch {epoch}/{total_epochs}",
        unit="batch",
        ncols=120,
    )

    for step, (images, labels) in enumerate(
        progress_bar,
        start=1,
    ):
        val_step_fn(images, labels)

        should_log = (
            step == 1
            or step % log_every_n_steps == 0
            or (
                total_steps is not None
                and step == total_steps
            )
        )

        if should_log:
            progress_bar.set_postfix(
                metric_snapshot(val_metrics),
                refresh=False,
            )


def save_json(data, path: str) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def train_worker() -> None:
    config = {
        # Dataset
        "train_dir": "src/data/train",
        "val_dir": "src/data/val",
        "test_dir": "src/data/test",

        # Image/data
        "image_size": 224,
        "image_channels": 3,
        "input_size": (224, 224),
        "input_shape": (224, 224, 3),
        "batch_size": 16,

        # Model
        "model_name": "efficientnet_b2",
        "pretrained": True,
        "weights": "imagenet",
        "freeze_backbone": False,
        "dropout": 0.2,
        "hidden_dim": 512,

        # Training
        "epochs": 50,
        "loss_name": "cross_entropy",
        "learning_rate": 1e-5,
        "min_learning_rate": 1e-7,
        "lr_factor": 0.3,
        "lr_patience": 3,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-4,

        # Performance
        "use_mixed_precision": True,
        "use_xla": False,
        "log_every_n_steps": 25,
        "val_log_every_n_steps": 50,

        # None = memory growth.
        # Use e.g. 6500 only if you explicitly want a hard VRAM limit.
        "gpu_memory_limit_mb": None,

        # Output/logging
        "output_dir": "src/outputs",
        "log_dir": "src/logs",

        # Reproducibility
        "seed": 42,
    }

    # Set Python/NumPy/TensorFlow random seeds together.
    tf.keras.utils.set_random_seed(
        config["seed"]
    )

    # IMPORTANT: runtime/GPU configuration must happen before model execution.
    setup_runtime(
        use_mixed_precision=config[
            "use_mixed_precision"
        ],
        gpu_memory_limit_mb=config[
            "gpu_memory_limit_mb"
        ],
    )

    os.makedirs(
        config["output_dir"],
        exist_ok=True,
    )
    os.makedirs(
        config["log_dir"],
        exist_ok=True,
    )

    print("\nLoading datasets...")

    (
        train_ds,
        val_ds,
        test_ds,
        class_names,
        num_classes,
        class_weight,
        class_counts,
    ) = build_datasets(config)

    train_ds = optimize_dataset(
        train_ds,
        training=True,
    )
    val_ds = optimize_dataset(
        val_ds,
        training=False,
    )
    test_ds = optimize_dataset(
        test_ds,
        training=False,
    )

    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {dataset_cardinality(train_ds)}")
    print(f"Val batches: {dataset_cardinality(val_ds)}")
    print(f"Test batches: {dataset_cardinality(test_ds)}")

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

    # Explicit warm build. This is especially useful for subclassed models.
    dummy_input = tf.zeros(
        (
            1,
            config["image_size"],
            config["image_size"],
            config["image_channels"],
        ),
        dtype=tf.float32,
    )

    _ = model(
        dummy_input,
        training=False,
    )

    print(
        f"Model parameters: "
        f"{model.count_params():,}"
    )

    loss_fn = get_loss(
        config["loss_name"]
    )

    base_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"]
    )

    if config["use_mixed_precision"]:
        optimizer = (
            tf.keras.mixed_precision
            .LossScaleOptimizer(
                base_optimizer
            )
        )
    else:
        optimizer = base_optimizer

    # Let the LR scheduler control the underlying Adam optimizer directly.
    lr_scheduler = ManualReduceLROnPlateau(
        optimizer=base_optimizer,
        monitor="val_loss",
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        min_lr=config["min_learning_rate"],
        min_delta=1e-4,
        verbose=True,
    )

    train_metrics = create_train_metrics(
        num_classes
    )
    val_metrics = create_val_metrics(
        num_classes
    )

    train_step_fn = create_train_step(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_metrics=train_metrics,
        class_weights=class_weight,
        use_mixed_precision=config[
            "use_mixed_precision"
        ],
        use_xla=config["use_xla"],
    )

    val_step_fn = create_val_step(
        model=model,
        loss_fn=loss_fn,
        val_metrics=val_metrics,
        use_xla=config["use_xla"],
    )

    (
        writer,
        tensorboard_log_dir,
    ) = create_tensorboard_writer(
        log_dir=config["log_dir"]
    )

    print(
        f"TensorBoard log dir: "
        f"{tensorboard_log_dir}"
    )

    best_val_macro_f1 = float("-inf")
    counter = 0

    best_model_path = os.path.join(
        config["output_dir"],
        "best_model.weights.h5",
    )

    final_model_path = os.path.join(
        config["output_dir"],
        "final_model.weights.h5",
    )

    history = []

    save_json(
        class_names,
        os.path.join(
            config["output_dir"],
            "class_names.json",
        ),
    )

    save_json(
        config,
        os.path.join(
            config["output_dir"],
            "config.json",
        ),
    )

    print("\nStart training...")

    for epoch in range(
        1,
        config["epochs"] + 1,
    ):
        reset_metrics(train_metrics)
        reset_metrics(val_metrics)

        print(
            f"\nEpoch {epoch}/"
            f"{config['epochs']}"
        )

        train_one_epoch(
            train_ds=train_ds,
            train_step_fn=train_step_fn,
            train_metrics=train_metrics,
            epoch=epoch,
            total_epochs=config["epochs"],
            log_every_n_steps=config[
                "log_every_n_steps"
            ],
        )

        validate_one_epoch_fast(
            val_ds=val_ds,
            val_step_fn=val_step_fn,
            val_metrics=val_metrics,
            epoch=epoch,
            total_epochs=config["epochs"],
            log_every_n_steps=config[
                "val_log_every_n_steps"
            ],
        )

        train_results = get_metric_results(
            train_metrics
        )
        val_results = get_metric_results(
            val_metrics
        )

        lr_scheduler.step(
            val_results["loss"]
        )

        learning_rate = float(
            tf.keras.backend.get_value(
                base_optimizer.learning_rate
            )
        )

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
            "train_accuracy": train_results[
                "accuracy"
            ],
            "train_mae": train_results["mae"],
            "train_macro_f1": train_results.get(
                "macro_f1"
            ),
            "val_loss": val_results["loss"],
            "val_accuracy": val_results[
                "accuracy"
            ],
            "val_mae": val_results["mae"],
            "val_macro_f1": val_results.get(
                "macro_f1"
            ),
            "learning_rate": learning_rate,
        }

        history.append(epoch_log)

        print(
            f"train_loss: "
            f"{train_results['loss']:.4f} | "
            f"train_acc: "
            f"{train_results['accuracy']:.4f} | "
            f"train_mae: "
            f"{train_results['mae']:.4f} | "
            f"train_f1: "
            f"{train_results.get('macro_f1', 0.0):.4f} | "
            f"val_loss: "
            f"{val_results['loss']:.4f} | "
            f"val_acc: "
            f"{val_results['accuracy']:.4f} | "
            f"val_mae: "
            f"{val_results['mae']:.4f} | "
            f"val_f1: "
            f"{val_results.get('macro_f1', 0.0):.4f} | "
            f"lr: {learning_rate:.2e}"
        )

        current_val_macro_f1 = (
            val_results.get(
                "macro_f1",
                float("-inf"),
            )
        )

        improved = (
            current_val_macro_f1
            > best_val_macro_f1
            + config[
                "early_stopping_min_delta"
            ]
        )

        if improved:
            best_val_macro_f1 = (
                current_val_macro_f1
            )
            counter = 0

            model.save_weights(
                best_model_path
            )

            print(
                f"Best model saved to "
                f"{best_model_path} "
                f"with val_macro_f1: "
                f"{best_val_macro_f1:.4f}"
            )
        else:
            counter += 1

            print(
                "No improvement. "
                f"Patience: "
                f"{counter}/"
                f"{config['early_stopping_patience']}"
            )

        save_json(
            history,
            os.path.join(
                config["output_dir"],
                "training_history.json",
            ),
        )

        if (
            counter
            >= config[
                "early_stopping_patience"
            ]
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    # Make final_model.weights.h5 represent the best validation checkpoint,
    # because the inference service currently expects this filename.
    if os.path.exists(best_model_path):
        model.load_weights(best_model_path)

    model.save_weights(
        final_model_path
    )

    writer.flush()

    print("\nTraining finished.")
    print(
        "Best validation macro F1: "
        f"{best_val_macro_f1:.4f}"
    )
    print(
        f"Best model path: "
        f"{best_model_path}"
    )
    print(
        f"Final model path: "
        f"{final_model_path}"
    )
    print(
        f"TensorBoard logs: "
        f"{tensorboard_log_dir}"
    )


if __name__ == "__main__":
    train_worker()