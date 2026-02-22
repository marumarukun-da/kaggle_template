#!/bin/bash
set -e
python src/download.py competition_download || { echo "Competition download failed"; exit 1; }
