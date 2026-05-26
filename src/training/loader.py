import tensorflow as tf
from training.preprocessing import preprocess_eval, preprocess_train

def build_datasets(config):
    image_size = config["input_size"]
    batch_size = config["batch_size"]
    seed = config.get("seed", 42)
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["train_dir"],
        label_mode="categorical",
        labels="inferred",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["val_dir"],
        label_mode="categorical",
        labels="inferred",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config["test_dir"],
        label_mode="categorical",
        labels="inferred",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)

    autotune = tf.data.AUTOTUNE

    train_ds = (
        train_ds
        .map(preprocess_train, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    val_ds = (
        val_ds
        .map(preprocess_eval, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    test_ds = (
        test_ds
        .map(preprocess_eval, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    return train_ds, val_ds, test_ds, class_names, num_classes

