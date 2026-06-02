import tensorflow as tf
from pathlib import Path

def count_images_per_class(train_dir, class_names):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_counts = {}

    for class_name in class_names:
        class_dir = Path(train_dir) / class_name

        count = sum(
            1 for f in class_dir.rglob("*")
            if f.suffix.lower() in image_exts
        )

        class_counts[class_name] = count

    return class_counts


def build_class_weight(train_dir, class_names):
    class_counts = count_images_per_class(train_dir, class_names)

    total_samples = sum(class_counts.values())
    num_classes = len(class_names)

    class_weight_values = []

    for class_name in class_names:
        weight = total_samples / (num_classes * class_counts[class_name])
        class_weight_values.append(weight)

    class_weight_tensor = tf.constant(class_weight_values, dtype=tf.float32)

    print("\nClass counts:")
    for cls, count in class_counts.items():
        print(f"{cls:25s}: {count}")

    print("\nClass weights:")
    for idx, cls in enumerate(class_names):
        print(f"{idx:2d} | {cls:25s}: {class_weight_values[idx]:.4f}")

    return class_weight_tensor, class_counts

def build_datasets(config):
    with tf.device("/CPU:0"):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["train_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
            shuffle=True,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["val_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
            shuffle=False,
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["test_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
            shuffle=False,
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    class_weight, class_counts = build_class_weight(
        config["train_dir"],
        class_names
    )

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True

    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    test_ds = test_ds.with_options(options)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, num_classes, class_weight, class_counts