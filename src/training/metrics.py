import tensorflow as tf

class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self,num_classes, name="macro_f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )
    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        cm= tf.math.confusion_matrix(
            y_true, 
            y_pred, 
            num_classes= self.num_classes, 
            dtype= tf.float32  
        )
        
        self.confusion_matrix.assign_add(cm)
    def result(self):
        cm =self.confusion_matrix
        
        true_positive =tf.linalg.diag_part(cm)
        predicted_positive = tf.reduce_sum(cm, axis=0)
        actual_positive = tf.reduce_sum(cm, axis=1)
        
        precision = true_positive / (predicted_positive + tf.keras.backend.epsilon())
        recall = true_positive / (actual_positive + tf.keras.backend.epsilon())
        
        f1 = 2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        
        macro_f1 = tf.reduce_mean(f1)
        
        return macro_f1
    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))
        
def create_train_metrics(num_classes):
    return {
        "loss": tf.keras.metrics.Mean(name="train_loss"),
        "accuracy": tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
        "mae": tf.keras.metrics.MeanAbsoluteError(name="train_mae"),
        "macro_f1": MacroF1Score(num_classes=num_classes, name="train_macro_f1"),
    }


def create_val_metrics(num_classes):
    return {
        "loss": tf.keras.metrics.Mean(name="val_loss"),
        "accuracy": tf.keras.metrics.CategoricalAccuracy(name="val_accuracy"),
        "mae": tf.keras.metrics.MeanAbsoluteError(name="val_mae"),
        "macro_f1": MacroF1Score(num_classes=num_classes, name="val_macro_f1"),
    }


def reset_metrics(metrics_dict):
    for metric in metrics_dict.values():
        metric.reset_state()


def get_metric_results(metrics_dict):
    return {
        name: float(metric.result().numpy())
        for name, metric in metrics_dict.items()
    }