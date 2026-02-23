#!/bin/sh
set -e

# Usage: sh scripts/push_artifacts.sh <exp_name> [--version VER] [--update]

if [ -z "$1" ]; then
    echo "Error: Experiment name is required."
    echo "Usage: sh scripts/push_artifacts.sh <exp_name> [--version VER] [--update]"
    exit 1
fi

echo "Uploading artifacts..."
python src/upload.py artifacts --exp_name "$@"
echo "Done!"
