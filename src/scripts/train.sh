#!/bin/bash

set -e

echo "===================================="
echo "Starting test/evaluation pipeline..."
echo "===================================="

python -m src.training.main --mode test

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