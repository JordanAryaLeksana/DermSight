import tensorflow as tf

def build_datasets(config):
    # seed = config.get("seed", 42)

    with tf.device("/CPU:0"):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["train_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["val_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config["test_dir"],
            image_size=config["input_size"],
            batch_size=config["batch_size"],
            seed=config["seed"],
            label_mode="categorical",
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Keep preprocessing pipeline CPU-side too.
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True

    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)
    test_ds = test_ds.with_options(options)

    return train_ds, val_ds, test_ds, class_names, num_classes