import argparse
import subprocess
import sys

from src.training.training import train_worker
from src.training.test import test_worker


def run_tensorboard(log_dir="logs", port=6006):
    command = [
        "tensorboard",
        "--logdir",
        log_dir,
        "--port",
        str(port),
    ]

    print(f"Starting TensorBoard at http://localhost:{port}")
    print(f"Log directory: {log_dir}")

    subprocess.run(command)


def main():
    parser = argparse.ArgumentParser(
        description="Main pipeline untuk training, testing, dan TensorBoard."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test", "full", "tensorboard"],
        help=(
            "Mode pipeline: "
            "'train' untuk training, "
            "'test' untuk evaluasi test set, "
            "'full' untuk training lalu testing, "
            "'tensorboard' untuk membuka dashboard TensorBoard."
        ),
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Folder log TensorBoard."
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port untuk TensorBoard."
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Running training pipeline...")
        train_worker()

    elif args.mode == "test":
        print("Running testing pipeline...")
        test_worker()

    elif args.mode == "full":
        print("Running full pipeline: training -> testing")
        train_worker()
        test_worker()

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()