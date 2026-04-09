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
: "${OMPTARGET_CC:=CC}"
: "${NVCC:=nvcc}"
: "${CUDA_PATH:=${CUDATOOLKIT_HOME:-}}"
: "${OMP_GPU_ARCH:=sm_80}"
: "${OMPTARGET_FLAVOR:=auto}"

#region agent log
debug_log "pre-fix" "H1" "scripts/build_polaris.sh:24" "compiler_selection" "OMPTARGET_CC=${OMPTARGET_CC}"
debug_log "pre-fix" "H2" "scripts/build_polaris.sh:25" "cwd_and_root" "PWD=${PWD};ROOT_DIR=${ROOT_DIR}"
#endregion

#region agent log
if [[ -z "${CUDA_PATH}" ]]; then
  if command -v "${NVCC}" >/dev/null 2>&1; then
    CUDA_PATH="$(dirname "$(dirname "$(command -v "${NVCC}")")")"
  fi
fi
debug_log "post-fix" "H6" "scripts/build_polaris.sh:34" "toolchain_paths" "CUDA_PATH=${CUDA_PATH};OMP_GPU_ARCH=${OMP_GPU_ARCH}"
debug_log "post-fix" "H7" "scripts/build_polaris.sh:35" "compiler_probe" "which_omptarget=$(command -v "${OMPTARGET_CC}" 2>/dev/null || echo missing)"
#endregion

omptarget_flags_nvhpc="-mp=gpu -gpu=cc80"
omptarget_flags_clang="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=${OMP_GPU_ARCH} --cuda-path=${CUDA_PATH}"
omptarget_flags_first=""
omptarget_flags_fallback=""
#region agent log
if [[ "${OMPTARGET_FLAVOR}" == "nvhpc" ]]; then
  omptarget_flags_first="${omptarget_flags_nvhpc}"
  omptarget_flags_fallback="${omptarget_flags_clang}"
elif [[ "${OMPTARGET_FLAVOR}" == "clang" ]]; then
  omptarget_flags_first="${omptarget_flags_clang}"
  omptarget_flags_fallback="${omptarget_flags_nvhpc}"
else
  # Auto mode: CC wrapper on Polaris commonly routes to nvc++, so try NVHPC first.
  if [[ "${OMPTARGET_CC}" == "CC" || "${OMPTARGET_CC}" == */CC ]]; then
    omptarget_flags_first="${omptarget_flags_nvhpc}"
    omptarget_flags_fallback="${omptarget_flags_clang}"
  else
    omptarget_flags_first="${omptarget_flags_clang}"
    omptarget_flags_fallback="${omptarget_flags_nvhpc}"
  fi
fi
debug_log "post-fix" "H10" "scripts/build_polaris.sh:64" "omptarget_flag_strategy" "flavor=${OMPTARGET_FLAVOR};first=${omptarget_flags_first};fallback=${omptarget_flags_fallback}"
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
err_file="${ROOT_DIR}/.omptarget_build_stderr.txt"
rm -f "${err_file}"
set +e
"${OMPTARGET_CC}" -O3 ${omptarget_flags_first} ${INC} \
  "${ROOT_DIR}/src/openmp_target_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openmp_target_gemm" 2> "${err_file}"
omptarget_rc=$?
if [[ "${omptarget_rc}" -ne 0 ]]; then
  if grep -qi "Unknown switch" "${err_file}" 2>/dev/null; then
    debug_log "post-fix" "H11" "scripts/build_polaris.sh:78" "omptarget_retry" "retry_with_fallback_flags"
    "${OMPTARGET_CC}" -O3 ${omptarget_flags_fallback} ${INC} \
      "${ROOT_DIR}/src/openmp_target_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openmp_target_gemm" 2> "${err_file}"
    omptarget_rc=$?
  fi
fi
set -e
debug_log "post-fix" "H4" "scripts/build_polaris.sh:51" "omp_target_build_rc" "rc=${omptarget_rc}"
if [[ "${omptarget_rc}" -ne 0 ]]; then
  if [[ -f "${err_file}" ]]; then
    first_err="$(head -n 1 "${err_file}" | tr '"' "'" | tr -d '\r' | tr -d '\n')"
    second_err="$(sed -n '2p' "${err_file}" | tr '"' "'" | tr -d '\r' | tr -d '\n')"
    debug_log "post-fix" "H8" "scripts/build_polaris.sh:57" "omp_target_first_error_line" "${first_err}"
    debug_log "post-fix" "H9" "scripts/build_polaris.sh:58" "omp_target_second_error_line" "${second_err}"
  fi
  debug_log "post-fix" "H5" "scripts/build_polaris.sh:53" "omp_target_build_failed" "likely_cuda_libdevice_or_arch_resolution_failure"
  exit "${omptarget_rc}"
fi
#endregion

echo "[build] CUDA -> ${BIN_DIR}/cuda_gemm"
#region agent log
cuda_err_file="${ROOT_DIR}/.cuda_build_stderr.txt"
rm -f "${cuda_err_file}"
set +e
"${NVCC}" -O3 ${INC} "${ROOT_DIR}/src/cuda_gemm.cu" "${COMMON_SRC}" -o "${BIN_DIR}/cuda_gemm" 2> "${cuda_err_file}"
cuda_rc=$?
set -e
debug_log "post-fix" "H12" "scripts/build_polaris.sh:111" "cuda_build_rc" "rc=${cuda_rc}"
if [[ "${cuda_rc}" -ne 0 ]]; then
  if [[ -f "${cuda_err_file}" ]]; then
    c1="$(head -n 1 "${cuda_err_file}" | tr '"' "'" | tr -d '\r' | tr -d '\n')"
    c2="$(sed -n '2p' "${cuda_err_file}" | tr '"' "'" | tr -d '\r' | tr -d '\n')"
    c3="$(sed -n '3p' "${cuda_err_file}" | tr '"' "'" | tr -d '\r' | tr -d '\n')"
    debug_log "post-fix" "H13" "scripts/build_polaris.sh:117" "cuda_stderr_line1" "${c1}"
    debug_log "post-fix" "H14" "scripts/build_polaris.sh:118" "cuda_stderr_line2" "${c2}"
    debug_log "post-fix" "H15" "scripts/build_polaris.sh:119" "cuda_stderr_line3" "${c3}"
  fi
  exit "${cuda_rc}"
fi
#endregion

echo "[build] done"
