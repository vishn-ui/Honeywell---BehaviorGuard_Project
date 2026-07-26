#!/usr/bin/env bash
# BehaviorGuard - build the whole pipeline from scratch, end to end.
set -e
cd "$(dirname "$0")"

echo "== 1/6 Installing dependencies =="
pip install -r requirements.txt --break-system-packages -q

echo "== 2/6 Generating synthetic behavioral data =="
cd data
python3 generate_data.py --n_entities 300 --days 45 --anomaly_rate 0.0025 \
    --out_sessions sessions.csv --out_entities entities.csv

echo "== 3/6 Building baseline profiles + features =="
cd ../src
python3 features.py ../data/sessions.csv ../data/features.csv

echo "== 4/6 Building sequences =="
python3 dataset.py

echo "== 5/6 Training the sequence-aware detector + classifier =="
python3 train.py

echo "== 6/6 Scoring test events -> ranked alerts =="
python3 infer.py

cd ../report
python3 generate_report.py

echo ""
echo "Done. To view the analyst dashboard, run:"
echo "    cd dashboard && streamlit run app.py"
