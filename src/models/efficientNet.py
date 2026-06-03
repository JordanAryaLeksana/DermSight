import tensorflow as tf
from tensorflow.keras import layers


class EfficientNetSkinClassifier(tf.keras.Model):

    SUPPORTED_BACKBONES = {
        "efficientnet_b0": tf.keras.applications.EfficientNetB0,
        "efficientnet_b1": tf.keras.applications.EfficientNetB1,
        "efficientnet_b2": tf.keras.applications.EfficientNetB2,
        "efficientnet_b3": tf.keras.applications.EfficientNetB3,
    }

    DEFAULT_INPUT_SIZES = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 224,
        "efficientnet_b2": 224,
        "efficientnet_b3": 224,
    }

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        image_channels: int = 3,
        input_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_backbone: bool = True,
        hidden_dim: int = 512,
        name: str = "efficientnet_skin_classifier",
    ):
        super().__init__(name=name)

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.image_channels = image_channels
        self.input_size = input_size
        self.pretrained = pretrained
        self.dropout_rate = dropout
        self.freeze_backbone = freeze_backbone
        self.hidden_dim = hidden_dim

        weights = "imagenet" if pretrained else None
        backbone_class = self.SUPPORTED_BACKBONES[backbone]

    
        if image_channels != 3:
            self.channel_adapter = layers.Conv2D(
                filters=3,
                kernel_size=1,
                padding="same",
                name="channel_adapter",
            )
            backbone_input_shape = (input_size, input_size, 3)
        else:
            self.channel_adapter = None
            backbone_input_shape = (input_size, input_size, 3)

        self.backbone = backbone_class(
            include_top=False,
            weights=weights,
            input_shape=backbone_input_shape,
        )

        self.backbone.trainable = not freeze_backbone

        self.data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal", name="aug_flip"),
                layers.RandomRotation(0.06, name="aug_rotation"),
                layers.RandomZoom(0.10, name="aug_zoom"),
                layers.RandomTranslation(0.05, 0.05, name="aug_translation"),
                layers.RandomContrast(0.12, name="aug_contrast"),
            ],
            name="data_augmentation",
        )

        self.pool = layers.GlobalAveragePooling2D(name="global_average_pooling")

        self.shared_fc = tf.keras.Sequential(
            [
                layers.BatchNormalization(name="shared_bn_1"),
                layers.Dropout(dropout, name="shared_dropout_1"),

                layers.Dense(hidden_dim, activation="relu", name="shared_dense"),
                layers.BatchNormalization(name="shared_bn_2"),
                layers.Dropout(0.3, name="shared_dropout_2"),
            ],
            name="shared_fc",
        )

        self.classifier_head = tf.keras.Sequential(
            [
                layers.Dense(
                    num_classes,
                    activation="softmax",
                    dtype="float32",
                  
                )
            ],
        )
    def extract_features(self, x, training=False):
        
        if training:
            x = self.data_augmentation(x, training=training)
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Saat backbone freeze, training=False agar BatchNorm tidak update.
        backbone_training = training if self.backbone.trainable else False

        x = self.backbone(x, training=backbone_training)
        x = self.pool(x)
        x = self.shared_fc(x, training=training)

        return x

    def call(self, x, training=False, return_features=False):
        features = self.extract_features(x, training=training)
        outputs = self.classifier_head(features, training=training)

        if return_features:
            return outputs, features

        return outputs

    def freeze_features(self):
        self.backbone.trainable = False

    def unfreeze_features(self):
        self.backbone.trainable = True

    def unfreeze_last_n_layers(self, n: int = 20):
        self.backbone.trainable = True

        for layer in self.backbone.layers[:-n]:
            layer.trainable = False

        for layer in self.backbone.layers[-n:]:
            layer.trainable = True


def build_efficientnet_classifier(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    image_channels: int = 3,
    input_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_backbone: bool = True,
    hidden_dim: int = 512,
) -> tf.keras.Model:

    model = EfficientNetSkinClassifier(
        num_classes=num_classes,
        backbone=backbone,
        image_channels=image_channels,
        input_size=input_size,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )

    dummy_input = tf.zeros(
        (1, input_size, input_size, image_channels),
        dtype=tf.float32,
    )
    _ = model(dummy_input, training=False)

    return model


def count_trainable_parameters(model: tf.keras.Model) -> int:
    return int(
        sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    )


def count_total_parameters(model: tf.keras.Model) -> int:
    return int(
        sum(tf.keras.backend.count_params(weight) for weight in model.weights)
    )