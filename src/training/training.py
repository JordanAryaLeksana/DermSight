import os 
import json
import tensorflow as tf

import os
import json
import tensorflow as tf

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

def setup_gpu(use_mixed_precision:bool = True):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU DETECTED: {gpus}")
    
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Could not set memory growth: {e}")
        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled: mixed_float16")
            except Exception as e:
                print(f"Could not enable mixed precision: {e}")
    else:
        print("No GPU detected. Training will use CPU.")


def train_one_epoch(
    model,
    train_ds, 
    loss_fn,
    optimizer, 
    train_metrics 
):
    for step, (images, labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            predictions = model(images, Training= True)
            
            loss = loss_fn(labels, predictions)
            loss = tf.reduce_mean()
            
        gradients = tape.gradient(loss, model.trainable_variables)
        
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )
        train_metrics["loss"].update_state(loss)
        train_metrics["accuracy"].update_state(labels, predictions)
        train_metrics["mae"].update_state(labels, predictions)

        if "macro_f1" in train_metrics:
            train_metrics["macro_f1"].update_state(labels, predictions)    

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        
def train_worker():
    config = {
        "train_dir": "src/data/train",
        "val_dir": "src/data/val",
        "test_dir": "src/data/test",

        "model_name": "efficientnet_b0",
        "image_size": 224,
        "image_channels": 3,
        "batch_size": 32,

        "epochs": 50,

        "pretrained": True,
        "dropout": 0.4,
        "freeze_backbone": True,
        "hidden_dim": 512,

        "loss_name": "focal_loss",

        "output_dir": "outputs",
        "log_dir": "logs",

        "seed": 42,

        # Untuk RTX GPU, mixed precision biasanya membantu.
        # Kalau muncul error numerik / loss NaN, ubah ke False.
        "use_mixed_precision": True,

        "early_stopping_patience": 10,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-7,
        "lr_factor": 0.3,
        "lr_patience": 2,
    }
     
    tf.random.set_seed(seed=config["seed"])
    
    setup_gpu(
        use_mixed_precision=config["use_mixed_precision"]
    )
    
    
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    print("\n loading datasets")
    
    train_ds, val_ds, test_ds, class_names, num_classes = build_datasets(config)
    
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

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"]
    )

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

    writer, tensorboard_log_dir = create_tensorboard_writer(
        log_dir=config["log_dir"]
    )

    print(f"TensorBoard log dir: {tensorboard_log_dir}")

    best_val_macro_f1 = 0.0
    patience_counter = 0

    best_model_path = os.path.join(
        config["output_dir"],
        "best_model.keras"
    )

    final_model_path = os.path.join(
        config["output_dir"],
        "final_model.keras"
    )

    history = []

    save_json(
        class_names,
        os.path.join(config["output_dir"], "class_names.json")
    )

    save_json(
        config,
        os.path.join(config["output_dir"], "config.json")
    )

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
            train_metrics=train_metrics
        )
        
        validate_one_epoch(
            model=model,
            val_ds=val_ds,
            loss_fn=loss_fn, 
            val_metrics=val_metrics,
        )
        
        train_results = get_metric_results(train_metrics)
        val_results = get_metric_results(val_metrics)

        lr_scheduler.step(val_results["loss"])

        learning_rate = float(
            tf.keras.backend.get_value(optimizer.learning_rate)
        )
        
        write_epoch_logs(
            writer=writer,
            epoch=epoch,
            train_results=train_results,
            val_results=val_results, 
            learning_rate=learning_rate
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
            
            model.save(best_model_path)
            print(
                f"Best model saved to {best_model_path} "
                f"with val_macro_f1: {best_val_macro_f1:.4f}"
            )
        else:
            counter += 1
            print(
                f"No improvement. "
                f"Patience: {patience_counter}/{config['early_stopping_patience']}"
            )
        save_json(
            history,
            os.path.join(config["output_dir"], "training_history.json")
        )

        if patience_counter >= config["early_stopping_patience"]:
            print("\nEarly stopping triggered.")
            break

    model.save(final_model_path)

    print("\nTraining finished.")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Best model path: {best_model_path}")
    print(f"Final model path: {final_model_path}")
    print(f"TensorBoard logs: {tensorboard_log_dir}")


if __name__ == "__main__":
    train_worker()