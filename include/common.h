#ifndef GEMM_COMMON_H
#define GEMM_COMMON_H

#include <stddef.h>

typedef struct {
    int n;
    int iters;
    int warmup;
    int repeat_id;
    unsigned int seed;
} GemmArgs;

typedef struct {
    double end_to_end_s;
    double h2d_s;
    double compute_s;
    double d2h_s;
    double gflops_compute;
    float max_abs_error;
} GemmMetrics;

int parse_gemm_args(int argc, char **argv, GemmArgs *args);
double wall_seconds(void);
void init_matrix_fp32(float *m, int n, unsigned int seed);
void zero_matrix_fp32(float *m, int n);
void gemm_reference_fp32(const float *a, const float *b, float *c, int n);
float max_abs_error_fp32(const float *a, const float *b, int n);
double compute_gflops(int n, double compute_s);
void print_csv_line(
    const char *version,
    int n,
    int repeat_id,
    const GemmMetrics *metrics
);

#endif
