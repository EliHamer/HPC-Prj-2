#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int parse_int_flag(const char *arg, const char *flag, int *dst) {
    size_t len = strlen(flag);
    if (strncmp(arg, flag, len) == 0 && arg[len] == '=') {
        *dst = atoi(arg + len + 1);
        return 1;
    }
    return 0;
}

static int parse_uint_flag(const char *arg, const char *flag, unsigned int *dst) {
    size_t len = strlen(flag);
    if (strncmp(arg, flag, len) == 0 && arg[len] == '=') {
        *dst = (unsigned int)strtoul(arg + len + 1, NULL, 10);
        return 1;
    }
    return 0;
}

int parse_gemm_args(int argc, char **argv, GemmArgs *args) {
    int i;
    args->n = 1024;
    args->iters = 5;
    args->warmup = 1;
    args->repeat_id = 0;
    args->seed = 12345u;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return 1;
        }
        if (parse_int_flag(argv[i], "--n", &args->n)) {
            continue;
        }
        if (parse_int_flag(argv[i], "--iters", &args->iters)) {
            continue;
        }
        if (parse_int_flag(argv[i], "--warmup", &args->warmup)) {
            continue;
        }
        if (parse_int_flag(argv[i], "--repeat-id", &args->repeat_id)) {
            continue;
        }
        if (parse_uint_flag(argv[i], "--seed", &args->seed)) {
            continue;
        }
        fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        return -1;
    }

    if (args->n <= 0 || args->iters <= 0 || args->warmup < 0) {
        fprintf(stderr, "Invalid args: n>0, iters>0, warmup>=0 required.\n");
        return -1;
    }
    return 0;
}

double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void init_matrix_fp32(float *m, int n, unsigned int seed) {
    int i;
    unsigned int x = seed;
    int total = n * n;
    for (i = 0; i < total; ++i) {
        x = 1664525u * x + 1013904223u;
        m[i] = (float)((x & 0x00FFFFFFu) / 16777216.0) - 0.5f;
    }
}

void zero_matrix_fp32(float *m, int n) {
    memset(m, 0, (size_t)n * (size_t)n * sizeof(float));
}

void gemm_reference_fp32(const float *a, const float *b, float *c, int n) {
    int i, j, k;
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

float max_abs_error_fp32(const float *a, const float *b, int n) {
    int i;
    float max_err = 0.0f;
    int total = n * n;
    for (i = 0; i < total; ++i) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

double compute_gflops(int n, double compute_s) {
    double ops = 2.0 * (double)n * (double)n * (double)n;
    if (compute_s <= 0.0) {
        return 0.0;
    }
    return ops / (compute_s * 1e9);
}

void print_csv_line(
    const char *version,
    int n,
    int repeat_id,
    const GemmMetrics *metrics
) {
    printf(
        "%s,%d,%d,%.9f,%.9f,%.9f,%.9f,%.6f,%.8e\n",
        version,
        n,
        repeat_id,
        metrics->end_to_end_s,
        metrics->h2d_s,
        metrics->compute_s,
        metrics->d2h_s,
        metrics->gflops_compute,
        metrics->max_abs_error
    );
}
