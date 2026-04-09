#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
mkdir -p "${BIN_DIR}"
DEBUG_LOG_PATH="${ROOT_DIR}/debug-41c8b9.log"

#region agent log
debug_log() {
  local run_id="$1"
  local hypothesis_id="$2"
  local location="$3"
  local message="$4"
  local data="$5"
  local ts
  ts="$(date +%s%3N 2>/dev/null || date +%s000)"
  printf '{"sessionId":"41c8b9","runId":"%s","hypothesisId":"%s","location":"%s","message":"%s","data":"%s","timestamp":%s}\n' \
    "$run_id" "$hypothesis_id" "$location" "$message" "$data" "$ts" >> "${DEBUG_LOG_PATH}"
}
#endregion

# Toolchain overrides (set these in your PBS script/environment as needed)
: "${CPU_CC:=cc}"
: "${OPENACC_CC:=nvc}"
: "${OMPTARGET_CC:=clang}"
: "${NVCC:=nvcc}"

#region agent log
debug_log "pre-fix" "H1" "scripts/build_polaris.sh:24" "compiler_selection" "OMPTARGET_CC=${OMPTARGET_CC}"
debug_log "pre-fix" "H2" "scripts/build_polaris.sh:25" "cwd_and_root" "PWD=${PWD};ROOT_DIR=${ROOT_DIR}"
#endregion

COMMON_SRC="${ROOT_DIR}/src/common.c"
INC="-I${ROOT_DIR}/include"

echo "[build] CPU OpenMP -> ${BIN_DIR}/cpu_openmp"
"${CPU_CC}" -O3 -fopenmp ${INC} "${ROOT_DIR}/src/cpu_openmp.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/cpu_openmp"

echo "[build] OpenACC -> ${BIN_DIR}/openacc_gemm"
"${OPENACC_CC}" -O3 -acc -Minfo=accel ${INC} "${ROOT_DIR}/src/openacc_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openacc_gemm"

echo "[build] OpenMP target -> ${BIN_DIR}/openmp_target_gemm"
#region agent log
debug_log "pre-fix" "H3" "scripts/build_polaris.sh:34" "omp_target_build_start" "target=nvptx64-nvidia-cuda"
set +e
"${OMPTARGET_CC}" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda ${INC} \
  "${ROOT_DIR}/src/openmp_target_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openmp_target_gemm"
omptarget_rc=$?
set -e
debug_log "pre-fix" "H4" "scripts/build_polaris.sh:40" "omp_target_build_rc" "rc=${omptarget_rc}"
if [[ "${omptarget_rc}" -ne 0 ]]; then
  debug_log "pre-fix" "H5" "scripts/build_polaris.sh:42" "omp_target_build_failed" "likely_cuda_libdevice_or_arch_resolution_failure"
  exit "${omptarget_rc}"
fi
#endregion

echo "[build] CUDA -> ${BIN_DIR}/cuda_gemm"
"${NVCC}" -O3 ${INC} "${ROOT_DIR}/src/cuda_gemm.cu" "${COMMON_SRC}" -o "${BIN_DIR}/cuda_gemm"

echo "[build] done"
