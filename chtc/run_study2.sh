#!/bin/bash
# =============================================================================
# HTCondor executable: Study 2 BioViL-T inference (runs inside container_image).
#
# Expected layout after extracting the data tarball (see stage_data.sh):
#   data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv
#   data/MIMIC-CXR-JPG/files/...
#
# Arguments (match study2.sub macros):
#   $1  Local name of transferred dataset tarball (default: study2_data.tar.gz)
#   $2  Run suffix for output folder naming (default: study2_$Cluster or study2_local)
#   $3  Net id (informational; used in staging remap comments only)
# =============================================================================
set -euo pipefail

DATA_TAR="${1:-study2_data.tar.gz}"
RUN_SUFFIX="${2:-study2_${CLUSTER:-local}}"
NETID="${3:-user}"

ROOT="${_CONDOR_SCRATCH_DIR:-/workspace}"
cd "${ROOT}"

export PYTHONPATH="/workspace/code:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.hf}"
mkdir -p "${HF_HOME}"

if [[ -f "${DATA_TAR}" ]]; then
  echo "Extracting ${DATA_TAR} into ${ROOT}"
  tar -xzf "${DATA_TAR}" -C "${ROOT}"
else
  echo "ERROR: dataset tarball not found: ${DATA_TAR}" >&2
  exit 1
fi

METADATA="${ROOT}/data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv"
IMAGES="${ROOT}/data/MIMIC-CXR-JPG/files"
OUT_DIR="${ROOT}/study2_out/${RUN_SUFFIX}"

DEVICE="cuda"
if [[ "${STUDY2_DEVICE:-cuda}" != "cuda" ]]; then
  DEVICE="${STUDY2_DEVICE}"
fi

EXTRA=()
if [[ -n "${STUDY2_MAX_PAIRS:-}" ]]; then
  EXTRA+=(--max-pairs "${STUDY2_MAX_PAIRS}")
fi

python /workspace/code/study2/scripts/run_inference.py \
  --metadata-csv "${METADATA}" \
  --images-root "${IMAGES}" \
  --output-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --log-level INFO \
  "${EXTRA[@]}"

echo "Packing outputs for transfer back to submit/staging host"
tar -cf study2_bundle.tar -C "${ROOT}" "study2_out/${RUN_SUFFIX}"

echo "Done (${NETID}, suffix=${RUN_SUFFIX})"
