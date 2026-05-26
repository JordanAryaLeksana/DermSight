#!/bin/bash

set -e

LOG_DIR="logs"
PORT=6006

echo "===================================="
echo "Starting TensorBoard..."
echo "TensorBoard URL: http://localhost:${PORT}"
echo "===================================="

python -m src.training.main --mode tensorboard --log-dir ${LOG_DIR} --port ${PORT} &
TENSORBOARD_PID=$!

sleep 3

if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:${PORT}" > /dev/null 2>&1 &
fi

echo "===================================="
echo "Starting training..."
echo "===================================="

python -m src.training.main --mode train

echo "===================================="
echo "Training finished."
echo "TensorBoard masih berjalan di:"
echo "http://localhost:${PORT}"
echo "===================================="

echo "Tekan CTRL+C untuk stop TensorBoard."

wait $TENSORBOARD_PID