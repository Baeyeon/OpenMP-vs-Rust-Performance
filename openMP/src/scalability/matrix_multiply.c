// Matrix Multiply (C = A * B) -- OpenMP Scalability benchmark
// Optional light tiling (tile_bs > 0). A=1, B=2 => every C[i,j] should equal 2n.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>

// Portable 64-byte-aligned allocation: try aligned_alloc/posix_memalign, fallback to malloc
static void* alloc64(size_t nbytes) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    size_t aligned = (nbytes + 63u) & ~((size_t)63u);
    void* p = aligned_alloc(64, aligned);
    if (!p) p = malloc(nbytes);
    return p;
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void* p = NULL;
    if (posix_memalign(&p, 64, nbytes) != 0) p = NULL;
    if (!p) p = malloc(nbytes);
    return p;
#else
    return malloc(nbytes);
#endif
}

static void init_ones(double *A, double *B, int n) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n*n; ++i) {
        A[i] = 1.0; B[i] = 2.0;
    }
}

static void zero_matrix(double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n*n; ++i) C[i] = 0.0;
}

static void mm_naive(double *A, double *B, double *C, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[(long long)i*n + k] * B[(long long)k*n + j];
            }
            C[(long long)i*n + j] += sum;
        }
    }
}

static void mm_tiled(double *A, double *B, double *C, int n, int bs) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += bs) {
        for (int jj = 0; jj < n; jj += bs) {
            for (int kk = 0; kk < n; kk += bs) {
                int iimax = (ii + bs < n) ? (ii + bs) : n;
                int jjmax = (jj + bs < n) ? (jj + bs) : n;
                int kkmax = (kk + bs < n) ? (kk + bs) : n;
                for (int i = ii; i < iimax; ++i) {
                    for (int j = jj; j < jjmax; ++j) {
                        double sum = 0.0;
                        for (int k = kk; k < kkmax; ++k) {
                            sum += A[(long long)i*n + k] * B[(long long)k*n + j];
                        }
                        C[(long long)i*n + j] += sum;
                    }
                }
            }
        }
    }
}

// Correctness: check all C entries are close to `target`
static int check_all_equal(const double *C, int n, double target, double tol) {
    int ok = 1;
    #pragma omp parallel
    {
        int local_ok = 1;
        #pragma omp for schedule(static)
        for (long long idx = 0; idx < (long long)n*n; ++idx) {
            if (!local_ok) continue;
            if (fabs(C[idx] - target) > tol) local_ok = 0;
        }
        if (!local_ok) {
            #pragma omp critical
            { ok = 0; }
        }
    }
    return ok;
}

static double median_of(double *arr, int m) {
    for (int i = 1; i < m; i++) {
        double v = arr[i]; int j = i - 1;
        while (j >= 0 && arr[j] > v) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = v;
    }
    return arr[m/2];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <n> <T> [tile_bs=0] [reps=5]\n", argv[0]);
        return 1;
    }
    const int n    = atoi(argv[1]);
    const int T    = atoi(argv[2]);
    const int bs   = (argc >= 4) ? atoi(argv[3]) : 0;
    const int reps = (argc >= 5) ? atoi(argv[4]) : 5;

    omp_set_num_threads(T);

    size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *A = (double*) alloc64(bytes);
    double *B = (double*) alloc64(bytes);
    double *C = (double*) alloc64(bytes);
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 2; }

    init_ones(A, B, n);
    zero_matrix(C, n);

    // Warm-up (not timed)
    if (bs > 0) mm_tiled(A, B, C, n, bs); else mm_naive(A, B, C, n);
    zero_matrix(C, n);

    // Repeat and take median time
    double *times = (double*) malloc(sizeof(double) * reps);
    for (int r = 0; r < reps; ++r) {
        zero_matrix(C, n);
        double t0 = omp_get_wtime();
        if (bs > 0) mm_tiled(A, B, C, n, bs); else mm_naive(A, B, C, n);
        double t1 = omp_get_wtime();
        times[r] = t1 - t0;
    }
    double t_med = median_of(times, reps);

    int ok = check_all_equal(C, n, 2.0 * (double)n, 1e-9);

    // CSV-style output:
    // time: median run time (sec) for the core multiplication
    // correct: 1 if A=1, B=2 produced C==2n everywhere
    printf("mm,openmp,%d,%d,time,%.6f,sec\n", n, T, t_med);
    printf("mm,openmp,%d,%d,correct,%d,boolean\n", n, T, ok);

    free(times); free(C); free(B); free(A);
    return ok ? 0 : 3;
}
