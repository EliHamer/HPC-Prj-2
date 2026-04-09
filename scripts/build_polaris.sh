#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
mkdir -p "${BIN_DIR}"

# Toolchain overrides (set these in your PBS script/environment as needed)
: "${CPU_CC:=cc}"
: "${OPENACC_CC:=nvc}"
: "${OMPTARGET_CC:=clang}"
: "${NVCC:=nvcc}"

COMMON_SRC="${ROOT_DIR}/src/common.c"
INC="-I${ROOT_DIR}/include"

echo "[build] CPU OpenMP -> ${BIN_DIR}/cpu_openmp"
"${CPU_CC}" -O3 -fopenmp ${INC} "${ROOT_DIR}/src/cpu_openmp.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/cpu_openmp"

echo "[build] OpenACC -> ${BIN_DIR}/openacc_gemm"
"${OPENACC_CC}" -O3 -acc -Minfo=accel ${INC} "${ROOT_DIR}/src/openacc_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openacc_gemm"

echo "[build] OpenMP target -> ${BIN_DIR}/openmp_target_gemm"
"${OMPTARGET_CC}" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda ${INC} \
  "${ROOT_DIR}/src/openmp_target_gemm.c" "${COMMON_SRC}" -lm -o "${BIN_DIR}/openmp_target_gemm"

echo "[build] CUDA -> ${BIN_DIR}/cuda_gemm"
"${NVCC}" -O3 ${INC} "${ROOT_DIR}/src/cuda_gemm.cu" "${COMMON_SRC}" -o "${BIN_DIR}/cuda_gemm"

echo "[build] done"
