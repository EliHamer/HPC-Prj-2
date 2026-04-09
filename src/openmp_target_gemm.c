#include "common.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void gemm_omp_target_kernel(const float *a, const float *b, float *c, int n) {
    int i, j, k;
    #pragma omp target teams distribute parallel for collapse(2) map(present, to: a[0:n*n], b[0:n*n]) map(present, from: c[0:n*n])
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    GemmArgs args;
    GemmMetrics m;
    size_t bytes;
    float *a, *b, *c, *ref;
    double end_sum = 0.0, h2d_sum = 0.0, comp_sum = 0.0, d2h_sum = 0.0;
    int rep;

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

    for (rep = 0; rep < args.warmup; ++rep) {
        #pragma omp target data map(to: a[0:args.n*args.n], b[0:args.n*args.n]) map(from: c[0:args.n*args.n])
        {
            gemm_omp_target_kernel(a, b, c, args.n);
        }
    }

    for (rep = 0; rep < args.iters; ++rep) {
        double t0 = wall_seconds();
        double th0 = wall_seconds();
        #pragma omp target enter data map(to: a[0:args.n*args.n], b[0:args.n*args.n], c[0:args.n*args.n])
        #pragma omp taskwait
        double th1 = wall_seconds();

        double tc0 = wall_seconds();
        gemm_omp_target_kernel(a, b, c, args.n);
        #pragma omp taskwait
        double tc1 = wall_seconds();

        double td0 = wall_seconds();
        #pragma omp target update from(c[0:args.n*args.n])
        #pragma omp taskwait
        double td1 = wall_seconds();
        #pragma omp target exit data map(delete: a[0:args.n*args.n], b[0:args.n*args.n], c[0:args.n*args.n])

        double t1 = wall_seconds();
        end_sum += (t1 - t0);
        h2d_sum += (th1 - th0);
        comp_sum += (tc1 - tc0);
        d2h_sum += (td1 - td0);
    }

    m.end_to_end_s = end_sum / args.iters;
    m.h2d_s = h2d_sum / args.iters;
    m.compute_s = comp_sum / args.iters;
    m.d2h_s = d2h_sum / args.iters;
    m.gflops_compute = compute_gflops(args.n, m.compute_s);
    m.max_abs_error = max_abs_error_fp32(c, ref, args.n);

    print_csv_line("openmp_target", args.n, args.repeat_id, &m);

    free(a);
    free(b);
    free(c);
    free(ref);
    return 0;
}
