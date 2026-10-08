import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.models.modelFactory import build_model


class SkinDiseasePredictor:
    def __init__(
        self,
        model_path: str,
        class_names_path: str,
        config_path: str,
    ):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.config_path = Path(config_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights file not found: "
                f"{self.model_path}"
            )

        if not self.class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found: "
                f"{self.class_names_path}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.class_names = self._load_class_names(
            self.class_names_path
        )

        if not self.class_names:
            raise ValueError(
                "class_names.json is empty or invalid"
            )

        self.image_size = int(
            self.config.get(
                "image_size",
                224,
            )
        )

        self.image_channels = int(
            self.config.get(
                "image_channels",
                3,
            )
        )

        self.model = build_model(
            model_name=self.config.get(
                "model_name",
                "efficientnet_b2",
            ),
            num_classes=len(
                self.class_names
            ),
            image_channels=self.image_channels,
            input_size=self.image_size,

            # Tidak perlu ImageNet karena seluruh
            # trained weights akan kita restore.
            pretrained=False,

            dropout=float(
                self.config.get(
                    "dropout",
                    0.4,
                )
            ),
            freeze_backbone=bool(
                self.config.get(
                    "freeze_backbone",
                    False,
                )
            ),
            hidden_dim=int(
                self.config.get(
                    "hidden_dim",
                    512,
                )
            ),
        )

        # Build all model variables first.
        dummy_input = tf.zeros(
            (
                1,
                self.image_size,
                self.image_size,
                self.image_channels,
            ),
            dtype=tf.float32,
        )

        _ = self.model(
            dummy_input,
            training=False,
        )

        # Restore ALL trained weights.
        self.model.load_weights(
            str(self.model_path)
        )

        self.model.trainable = False

    def _load_class_names(
        self,
        path: Path,
    ):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            try:
                return [
                    data[str(i)]
                    for i in range(len(data))
                ]
            except KeyError:
                pass

            if all(
                isinstance(v, int)
                for v in data.values()
            ):
                sorted_items = sorted(
                    data.items(),
                    key=lambda item: item[1],
                )

                return [
                    name
                    for name, _ in sorted_items
                ]

        raise ValueError(
            "Unsupported class_names.json format. "
            "Use list or index-label dictionary."
        )

    def _preprocess_image(
        self,
        image_bytes: bytes,
    ) -> np.ndarray:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            )
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        image_array = preprocess_input(
            image_array
        )

        return image_array

    def predict(
        self,
        image_bytes: bytes,
    ) -> Dict[str, Any]:
        if not image_bytes:
            raise ValueError(
                "Image bytes is empty"
            )

        input_tensor = (
            self._preprocess_image(
                image_bytes
            )
        )

        predictions = self.model(
            input_tensor,
            training=False,
        )

        probabilities = (
            predictions.numpy()[0]
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        if (
            predicted_index
            >= len(self.class_names)
        ):
            raise IndexError(
                f"Predicted index "
                f"{predicted_index} exceeds "
                f"class_names length "
                f"{len(self.class_names)}"
            )

        predicted_label = (
            self.class_names[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        return {
            "predicted_label":
                predicted_label,
            "confidence":
                round(confidence, 4),
        }