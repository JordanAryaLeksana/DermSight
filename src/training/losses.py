import tensorflow as tf


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name:str = "categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)   
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)   
        
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config
def get_loss(loss_name:str = "focal_loss"):
    loss_name = loss_name.lower()
    if loss_name == "focal_loss":
        return CategoricalFocalLoss()
    elif loss_name == "cross_entropy":
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}. Available losses: ['focal_loss', 'cross_entropy']")