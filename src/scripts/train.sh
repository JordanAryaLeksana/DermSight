#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

LOG_DIR="logs"
PORT=6006

echo "===================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Python path: ${PYTHONPATH}"
echo "Starting TensorBoard..."
echo "TensorBoard URL: http://localhost:${PORT}"
echo "===================================="

python "${PROJECT_ROOT}/src/main.py" --mode tensorboard --log-dir "${LOG_DIR}" --port "${PORT}" &
TENSORBOARD_PID=$!

sleep 3

if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:${PORT}" > /dev/null 2>&1 &
fi

echo "===================================="
echo "Starting training pipeline..."
echo "===================================="

python "${PROJECT_ROOT}/src/main.py" --mode train

echo "===================================="
echo "Training finished."
echo "TensorBoard still running at:"
echo "http://localhost:${PORT}"
echo "===================================="

echo "Press CTRL+C to stop TensorBoard."

wait $TENSORBOARD_PID