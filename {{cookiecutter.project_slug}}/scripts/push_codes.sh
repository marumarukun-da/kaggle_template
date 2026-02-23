#!/bin/sh
set -e

echo "Uploading codes dataset..."
python src/upload.py codes
echo "Done!"
