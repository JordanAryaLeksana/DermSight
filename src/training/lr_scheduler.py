# src/training/lr_scheduler.py

import tensorflow as tf


class ManualReduceLROnPlateau:


    def __init__(
        self,
        optimizer,
        monitor="val_loss",
        mode="min",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        min_delta=1e-4,
        verbose=True,
    ):
        self.optimizer = optimizer
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.verbose = verbose

        self.wait = 0

        if mode == "min":
            self.best = float("inf")
        elif mode == "max":
            self.best = -float("inf")
        else:
            raise ValueError("mode harus 'min' atau 'max'.")

    def _is_improvement(self, current):
        if self.mode == "min":
            return current < self.best - self.min_delta

        return current > self.best + self.min_delta

    def step(self, current_value):

        if self._is_improvement(current_value):
            self.best = current_value
            self.wait = 0
            return False

        self.wait += 1

        if self.wait >= self.patience:
            old_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
            new_lr = max(old_lr * self.factor, self.min_lr)

            tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)

            self.wait = 0

            if self.verbose:
                print(
                    f"Learning rate reduced: "
                    f"{old_lr:.8f} -> {new_lr:.8f}"
                )

            return True

        return False