#!/usr/bin/env bash
# =============================================================================
# Build a tarball of Study 2 inputs for HTCondor staging.
# Run on a machine that has the credentialed MIMIC exports (not from Windows
# unless you use WSL/Git Bash). From repository root:
#
#   bash chtc/stage_data.sh
#   bash chtc/stage_data.sh /staging/YOURNETID/study2_data.tar.gz
#
# Upload the tarball to CHTC staging (e.g. transfer.chtc.wisc.edu) under your
# netid, then point transfer_input_files in study2.sub at that path.
#
# The full MIMIC-CXR-JPG tree is large; stage only what you need or use
# --max-patients via a local dry run and a smaller subset tarball for tests.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${1:-${REPO_ROOT}/chtc/study2_data.tar.gz}"
META="data/MIMIC-CXR/csv/mimic-cxr-2.0.0-metadata.csv"
JPG="data/MIMIC-CXR-JPG/files"

if [[ ! -f "${META}" ]]; then
  echo "Missing ${META}" >&2
  exit 1
fi
if [[ ! -d "${JPG}" ]]; then
  echo "Missing ${JPG}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"
echo "Creating ${OUT} (metadata + JPG tree)"
tar -czf "${OUT}" \
  "${META}" \
  "${JPG}"

echo "OK: ${OUT}"
ls -lh "${OUT}"
