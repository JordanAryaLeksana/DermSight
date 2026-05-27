#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "===================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Python path: ${PYTHONPATH}"
echo "Starting test/evaluation pipeline..."
echo "===================================="

python "${PROJECT_ROOT}/src/main.py" --mode test

echo "===================================="
echo "Testing finished."
echo "Check output files in: outputs/"
echo "===================================="

echo "Generated files should include:"
echo "- outputs/test_metrics.json"
echo "- outputs/classification_report.json"
echo "- outputs/classification_report.txt"
echo "- outputs/test_predictions.csv"
echo "- outputs/confusion_matrix.csv"