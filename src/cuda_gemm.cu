#include "common.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ static void gemm_kernel(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

static double elapsed_event_s(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return (double)ms * 1e-3;
}

int main(int argc, char **argv) {
    GemmArgs args;
    GemmMetrics m;
    size_t bytes;
    float *a, *b, *c, *ref;
    float *d_a, *d_b, *d_c;
    cudaEvent_t e0, e1, e2, e3;
    dim3 block(16, 16);
    double end_sum = 0.0, h2d_sum = 0.0, comp_sum = 0.0, d2h_sum = 0.0;

    int parse_rc = parse_gemm_args(argc, argv, &args);
    if (parse_rc != 0) {
        if (parse_rc > 0) {
            printf("Usage: %s --n=<size> --iters=<iters> --warmup=<warmups> --repeat-id=<id> --seed=<seed>\n", argv[0]);
            return 0;
        }
        return 1;
    }

    bytes = (size_t)args.n * (size_t)args.n * sizeof(float);
    a = (float *)malloc(bytes);
    b = (float *)malloc(bytes);
    c = (float *)malloc(bytes);
    ref = (float *)malloc(bytes);
    if (!a || !b || !c || !ref) {
        fprintf(stderr, "Allocation failed.\n");
        free(a); free(b); free(c); free(ref);
        return 1;
    }

    init_matrix_fp32(a, args.n, args.seed + 1u);
    init_matrix_fp32(b, args.n, args.seed + 2u);
    gemm_reference_fp32(a, b, ref, args.n);

    check_cuda(cudaMalloc((void **)&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc((void **)&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc((void **)&d_c, bytes), "cudaMalloc d_c");
    check_cuda(cudaEventCreate(&e0), "cudaEventCreate e0");
    check_cuda(cudaEventCreate(&e1), "cudaEventCreate e1");
    check_cuda(cudaEventCreate(&e2), "cudaEventCreate e2");
    check_cuda(cudaEventCreate(&e3), "cudaEventCreate e3");

    dim3 grid((unsigned int)((args.n + block.x - 1) / block.x), (unsigned int)((args.n + block.y - 1) / block.y));
    for (int rep = 0; rep < args.warmup; ++rep) {
        check_cuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "warmup memcpy a");
        check_cuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "warmup memcpy b");
        gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, args.n);
        check_cuda(cudaGetLastError(), "warmup kernel launch");
        check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "warmup memcpy c");
    }
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    for (int rep = 0; rep < args.iters; ++rep) {
        double t0 = wall_seconds();

        check_cuda(cudaEventRecord(e0), "record e0");
        check_cuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "memcpy a");
        check_cuda(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "memcpy b");
        check_cuda(cudaEventRecord(e1), "record e1");

        check_cuda(cudaEventRecord(e2), "record e2");
        gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, args.n);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaEventRecord(e3), "record e3");
        check_cuda(cudaEventSynchronize(e3), "sync e3");

        double td0 = wall_seconds();
        check_cuda(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "memcpy c");
        double td1 = wall_seconds();

        double t1 = wall_seconds();
        end_sum += (t1 - t0);
        h2d_sum += elapsed_event_s(e0, e1);
        comp_sum += elapsed_event_s(e2, e3);
        d2h_sum += (td1 - td0);
    }

    m.end_to_end_s = end_sum / args.iters;
    m.h2d_s = h2d_sum / args.iters;
    m.compute_s = comp_sum / args.iters;
    m.d2h_s = d2h_sum / args.iters;
    m.gflops_compute = compute_gflops(args.n, m.compute_s);
    m.max_abs_error = max_abs_error_fp32(c, ref, args.n);

    print_csv_line("cuda", args.n, args.repeat_id, &m);

    check_cuda(cudaEventDestroy(e0), "destroy e0");
    check_cuda(cudaEventDestroy(e1), "destroy e1");
    check_cuda(cudaEventDestroy(e2), "destroy e2");
    check_cuda(cudaEventDestroy(e3), "destroy e3");
    check_cuda(cudaFree(d_a), "free d_a");
    check_cuda(cudaFree(d_b), "free d_b");
    check_cuda(cudaFree(d_c), "free d_c");
    free(a);
    free(b);
    free(c);
    free(ref);
    return 0;
}
