import tensorflow as tf
import datetime
import os 


def create_tensorboard_writer(log_dir="logs"):
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(log_dir, curr_time)
    
    writer = tf.summary.create_file_writer(run_log_dir)
    
    return writer, run_log_dir

def write_epoch_logs(
    writer,
    epoch,
    train_results,
    val_results,
    learning_rate,
):
    """
    Menulis metric training dan validation ke TensorBoard.
    """

    with writer.as_default():
        tf.summary.scalar(
            "Loss/train",
            train_results["loss"],
            step=epoch,
        )

        tf.summary.scalar(
            "Loss/validation",
            val_results["loss"],
            step=epoch,
        )

        tf.summary.scalar(
            "Accuracy/train",
            train_results["accuracy"],
            step=epoch,
        )

        tf.summary.scalar(
            "Accuracy/validation",
            val_results["accuracy"],
            step=epoch,
        )

        tf.summary.scalar(
            "MAE/train",
            train_results["mae"],
            step=epoch,
        )

        tf.summary.scalar(
            "MAE/validation",
            val_results["mae"],
            step=epoch,
        )

        if "macro_f1" in train_results:
            tf.summary.scalar(
                "Macro_F1/train",
                train_results["macro_f1"],
                step=epoch,
            )

        if "macro_f1" in val_results:
            tf.summary.scalar(
                "Macro_F1/validation",
                val_results["macro_f1"],
                step=epoch,
            )

        tf.summary.scalar(
            "Learning_Rate",
            learning_rate,
            step=epoch,
        )

        writer.flush()