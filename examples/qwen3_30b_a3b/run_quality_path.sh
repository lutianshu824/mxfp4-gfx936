#!/usr/bin/env bash
set -euo pipefail

: "${MXFP4_MODEL:?set MXFP4_MODEL to a local checkpoint path}"
: "${MXFP4_RESULT_PATH:?set MXFP4_RESULT_PATH to an output JSON path}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MXFP4_SOFT_MODULE="${REPO_ROOT}/src/aiter/ops/triton/moe_op_mxfp4_soft.py"

python "${SCRIPT_DIR}/eval_qwen3_30b_moe_quality.py"
