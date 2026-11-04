#!/usr/bin/env bash
# Rebuild output/spareparts.db from the six raw Excel files in data/raw/
set -e
cd "$(dirname "$0")"
python3 01_cleaning.py
python3 02_forecast.py
python3 04_leadtime_nlp.py
python3 05_global_model.py
python3 06_optimize.py
python3 07_tam_readiness.py
python3 09_export.py
