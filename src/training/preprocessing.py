import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input

def get_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

def preprocess_train(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

def preprocess_eval(image, label):
    """
    Preprocessing untuk validation dan test.
    Tidak ada augmentation random.
    """
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label