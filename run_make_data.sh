#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# STC-Traj2vec generalized data-generation runner
#
# Usage:
#   ./run_make_data.sh geolife
#   ./run_make_data.sh porto
#
# Edit the INPUT_* and OUTPUT_ROOT paths below.
# ------------------------------------------------------------------

DATASET="${1:-}"

# Change these paths for your environment.
INPUT_GEOLIFE="/data/geolife.csv"
INPUT_PORTO="train.csv"
OUTPUT_ROOT="./processed"

if [[ -z "${DATASET}" ]]; then
    echo "Usage: $0 {geolife|porto}"
    exit 1
fi

case "${DATASET}" in
    geolife)
        INPUT="${INPUT_GEOLIFE}"
        OUTPUT="${OUTPUT_ROOT}/geolife"
        ;;
    porto)
        INPUT="${INPUT_PORTO}"
        OUTPUT="${OUTPUT_ROOT}/porto"
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        echo "Usage: $0 {geolife|porto}"
        exit 1
        ;;
esac

mkdir -p "${OUTPUT}"

echo "=============================================="
echo "STC-Traj2vec data preparation"
echo "Dataset : ${DATASET}"
echo "Input   : ${INPUT}"
echo "Output  : ${OUTPUT}"
echo "=============================================="

python make_data.py \
    --dataset "${DATASET}" \
    --input "${INPUT}" \
    --output "${OUTPUT}"

echo
echo "Finished: ${DATASET}"
