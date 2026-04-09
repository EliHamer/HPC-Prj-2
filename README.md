# Polaris GEMM 5-Way Benchmark

This project compares FP32 dense square GEMM (`C = A * B`) across five implementations:

- CPU OpenMP (`src/cpu_openmp.c`)
- OpenACC offload (`src/openacc_gemm.c`)
- OpenMP target offload (`src/openmp_target_gemm.c`)
- CUDA (`src/cuda_gemm.cu`)
- PyTorch GPU (`pytorch/torch_gemm.py`)

All implementations accept:

- `--n=<matrix_size>`
- `--iters=<measured_iterations>`
- `--warmup=<warmup_iterations>`
- `--repeat-id=<repeat_index>`
- `--seed=<seed>`

Output is one CSV row per run:

`version,n,repeat,end_to_end_s,h2d_s,compute_s,d2h_s,gflops_compute,max_abs_error`

## Build on Polaris

Adjust modules/toolchain as needed, then:

```bash
bash scripts/build_polaris.sh
```

## Run Full Sweep (Batch)

Edit allocation and queue in `scripts/run_sweep.pbs`, then submit:

```bash
qsub scripts/run_sweep.pbs
```

The batch script runs sizes:

- `256, 512, 1024, 2048, 4096`

with:

- `REPEATS=5`
- `ITERS=5`
- `WARMUP=1`

and writes:

- raw runs CSV under `results/raw/`
- summary CSV at `results/summary_by_version_size.csv`
- markdown report at `report/comparison.md`

## Manual Aggregation

```bash
python scripts/aggregate_results.py \
  --input results/raw/runs_<jobid>.csv \
  --summary-csv results/summary_by_version_size.csv \
  --report-md report/comparison.md
```
