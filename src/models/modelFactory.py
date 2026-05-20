from src.models.efficientnet import build_efficientnet_classifier


def build_model(
    model_name: str,
    num_classes: int,
    image_channels: int = 3,
    input_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_backbone: bool = True,
    hidden_dim: int = 512,
):
    """
    Model factory.

    Tujuannya agar script training tidak perlu tahu detail class model.
    Kalau nanti ingin tambah ResNet, MobileNet, DenseNet, cukup register di sini.
    """

    efficientnet_models = [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
    ]

    if model_name in efficientnet_models:
        return build_efficientnet_classifier(
            num_classes=num_classes,
            backbone=model_name,
            image_channels=image_channels,
            input_size=input_size,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            hidden_dim=hidden_dim,
        )

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Available models: {efficientnet_models}"
    )