// OpenMP Matrix Multiply Scalability Benchmark (no reps, single run per (n,T))
// A = 1, B = 2  => C[i,j] = 2 * n
// Output format imitates the Rust version:
//   === OpenMP Matrix Multiply Benchmark (Scalability) ===
//   Testing problem sizes: [...]
//   Testing thread counts: [...]
//   ------------------------------------------------------
//   Problem Size: n = 256
//   ...
//   Threads =  1 ... Time: xxxxs (baseline)
//   Threads =  2 ... Time: xxxxs, Speedup: xx.x, Efficiency: yy.yy%

// ------------ 64-byte aligned allocation ------------

// Initialize A=1, B=2
// (initialize matrices A and B with constant values)
static void init_ones(double *A, double *B, int n) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n*n; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
    }
}

// Set C to all zeros
static void zero_matrix(double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n*n; ++i) {
        C[i] = 0.0;
    }
}

// Naive matrix multiply: C = A * B
static void mm_naive(double *A, double *B, double *C, int n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[(long long)i*n + k] * B[(long long)k*n + j];
            }
            C[(long long)i*n + j] = sum;
        }
    }
}

// Correctness check: whether all elements of C are close to `target`
static int check_all_equal(const double *C, int n, double target, double tol) {
    int ok = 1;
    #pragma omp parallel
    {
        int local_ok = 1;
        #pragma omp for schedule(static)
        for (long long idx = 0; idx < (long long)n*n; ++idx) {
            if (!local_ok) continue;
            if (fabs(C[idx] - target) > tol) {
                local_ok = 0;
            }
        }
        if (!local_ok) {
            #pragma omp critical
            { ok = 0; }
        }
    }
    return ok;
}

int main(void) {
    // Problem size set and thread count set (same as Rust version)
    const int Ns[] = {256, 512, 1024, 1536, 2048};
    const int n_cnt = (int)(sizeof(Ns) / sizeof(Ns[0]));

    const int Ts[] = {1, 2, 4, 8, 16};
    const int t_cnt = (int)(sizeof(Ts) / sizeof(Ts[0]));

    // Top header
    printf("=== OpenMP Matrix Multiply Benchmark (Scalability) ===\n");
    printf("Testing problem sizes: [256, 512, 1024, 1536, 2048]\n");
    printf("Testing thread counts: [1, 2, 4, 8, 16]\n\n");

    for (int ni = 0; ni < n_cnt; ++ni) {
        int n = Ns[ni];

        printf("============================================================\n");
        printf("Problem Size: n = %d\n", n);
        printf("============================================================\n\n");

        size_t bytes = (size_t)n * (size_t)n * sizeof(double);
        double *A = (double*) alloc64(bytes);
        double *B = (double*) alloc64(bytes);
        double *C = (double*) alloc64(bytes);
        if (!A || !B || !C) {
            fprintf(stderr, "malloc failed for n=%d\n", n);
            return 2;
        }

        init_ones(A, B, n);
        zero_matrix(C, n);

        double t_base = -1.0;   // t(n,1)

        for (int ti = 0; ti < t_cnt; ++ti) {
            int T = Ts[ti];
            omp_set_num_threads(T);

            // Time a single run
            zero_matrix(C, n);
            double t0 = omp_get_wtime();
            mm_naive(A, B, C, n);
            double t1 = omp_get_wtime();
            double t = t1 - t0;

            // Correctness check for this run
            int ok = check_all_equal(C, n, 2.0 * (double)n, 1e-9);

            if (ti == 0) {
                // baseline: T = 1
                printf("Threads = %2d ... Time: %.6lfs (baseline)%s\n",
                       T, t, ok ? "" : "  [INCORRECT]");
                t_base = t;
            } else {
                double speedup    = t_base / t;
                double efficiency = (speedup / (double)T) * 100.0; // percentage

                printf("Threads = %2d ... Time: %.6lfs, "
                       "Speedup: %.2lfx, Efficiency: %.2lf%%%s\n",
                       T, t, speedup, efficiency,
                       ok ? "" : "  [INCORRECT]");
            }
            fflush(stdout);  // print progressively while running
        }

        printf("\n");  // blank line after each problem size

        free(C);
        free(B);
        free(A);
    }

    return 0;
}
