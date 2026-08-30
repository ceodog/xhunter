#!/usr/bin/env bash
# Per-task entrypoint for Google Batch. Batch sets BATCH_TASK_INDEX
# automatically (0-indexed) -- used here to give each task a
# non-overlapping seed range, so N tasks together produce a single
# coherent training set. Each task writes to its OWN subdirectory
# (task_XXXXX) because planetx.simgen.orchestrate.generate_dataset always
# numbers its shard files starting from shard_00000 -- without this,
# concurrent tasks writing to the same OUT_DIR would clobber each other's
# shard_00000.parquet.
set -euo pipefail

TASK_INDEX="${BATCH_TASK_INDEX:-0}"
N_SIMS_PER_TASK="${N_SIMS_PER_TASK:?set N_SIMS_PER_TASK}"
WORKERS="${WORKERS:-64}"
PRIOR_PATH="${PRIOR_PATH:-configs/prior.yaml}"
OUT_DIR="${OUT_DIR:?set OUT_DIR, e.g. /mnt/disks/gcs/run1}"

SEED0=$(( TASK_INDEX * N_SIMS_PER_TASK ))
TASK_OUT_DIR="${OUT_DIR}/task_$(printf '%05d' "${TASK_INDEX}")"

echo "task ${TASK_INDEX}: ${N_SIMS_PER_TASK} sims, seed0=${SEED0}, workers=${WORKERS} -> ${TASK_OUT_DIR}"

exec planetx simgen run \
    --prior "${PRIOR_PATH}" \
    --out "${TASK_OUT_DIR}" \
    --n-sims "${N_SIMS_PER_TASK}" \
    --shard-size "${N_SIMS_PER_TASK}" \
    --workers "${WORKERS}" \
    --seed0 "${SEED0}"
