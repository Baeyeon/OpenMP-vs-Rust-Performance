// Runtime Overhead benchmark (OpenMP)
// Measures:
//   1) Parallel region overhead: repeated `#pragma omp parallel {}`
//   2) Barrier overhead: single parallel region with repeated `#pragma omp barrier`
//   3) Critical section overhead: `#pragma omp critical` (mutex equivalent)
//   4) Atomic operations overhead: `#pragma omp atomic`
//
// Usage:
//   ./overhead_openmp <T> <R>
//   T = number of threads (e.g., 1,2,4,8,16)
//   R = number of repetitions (e.g., 100000)
//
// Output: CSV-style lines with unified units (ms for total, ns for per-op), e.g.:
//   overhead,openmp,T=8,R=100000,parallel_total,12.34,ms
//   overhead,openmp,T=8,R=100000,parallel_per,123.45,ns
//   overhead,openmp,T=8,R=100000,barrier_total,34.56,ms
//   overhead,openmp,T=8,R=100000,barrier_per,345.67,ns
//   overhead,openmp,T=8,R=100000,critical_total,45.67,ms
//   overhead,openmp,T=8,R=100000,critical_per,567.89,ns
//   overhead,openmp,T=8,R=100000,atomic_total,23.45,ms
//   overhead,openmp,T=8,R=100000,atomic_per,234.56,ns
//
// Note: All "per" values are per-operation costs normalized by (iterations * threads)

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <T> <R>\n", argv[0]);
        fprintf(stderr, "  T = number of threads (e.g. 1,2,4,8,16)\n");
        fprintf(stderr, "  R = number of repetitions (e.g. 100000)\n");
        return 1;
    }

    int T = atoi(argv[1]);       // thread count
    long long R = atoll(argv[2]); // repetitions

    if (T <= 0 || R <= 0) {
        fprintf(stderr, "T and R must be positive\n");
        return 1;
    }

    omp_set_num_threads(T);

    // Optional warm-up: create threads once so we don't count one-time cost too much
    #pragma omp parallel
    {
        // empty warm-up parallel region
    }

    // -------------------------------
    // Test 1: Parallel region overhead
    // -------------------------------
    double t0 = omp_get_wtime();
    for (long long r = 0; r < R; ++r) {
        #pragma omp parallel
        {
            // empty body: we only measure enter/exit overhead
        }
    }
    double t1 = omp_get_wtime();
    double time_parallel = t1 - t0;
    double per_parallel  = time_parallel / (double)R;

    // ----------------------------
    // Test 2: Barrier overhead
    // ----------------------------
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        for (long long r = 0; r < R; ++r) {
            #pragma omp barrier
        }
    }
    t1 = omp_get_wtime();
    double time_barrier = t1 - t0;
    double per_barrier  = time_barrier / (double)(R * T);

    // ----------------------------
    // Test 3: Critical section overhead (mutex equivalent)
    // ----------------------------
    long long counter = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        for (long long r = 0; r < R; ++r) {
            #pragma omp critical
            {
                counter++;
            }
        }
    }
    t1 = omp_get_wtime();
    double time_critical = t1 - t0;
    double per_critical = time_critical / (double)(R * T);

    // ----------------------------
    // Test 4: Atomic operations overhead
    // ----------------------------
    long long atomic_counter = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        for (long long r = 0; r < R; ++r) {
            #pragma omp atomic
            atomic_counter++;
        }
    }
    t1 = omp_get_wtime();
    double time_atomic = t1 - t0;
    double per_atomic = time_atomic / (double)(R * T);

    // Unified output format (milliseconds for total, nanoseconds for per-op)
    // Matches Rust output format for easy comparison
    printf("overhead,openmp,T=%d,R=%lld,parallel_total,%.2f,ms\n",
           T, R, time_parallel * 1000.0);
    printf("overhead,openmp,T=%d,R=%lld,parallel_per,%.2f,ns\n",
           T, R, per_parallel * 1e9);
    printf("overhead,openmp,T=%d,R=%lld,barrier_total,%.2f,ms\n",
           T, R, time_barrier * 1000.0);
    printf("overhead,openmp,T=%d,R=%lld,barrier_per,%.2f,ns\n",
           T, R, per_barrier * 1e9);
    printf("overhead,openmp,T=%d,R=%lld,critical_total,%.2f,ms\n",
           T, R, time_critical * 1000.0);
    printf("overhead,openmp,T=%d,R=%lld,critical_per,%.2f,ns\n",
           T, R, per_critical * 1e9);
    printf("overhead,openmp,T=%d,R=%lld,atomic_total,%.2f,ms\n",
           T, R, time_atomic * 1000.0);
    printf("overhead,openmp,T=%d,R=%lld,atomic_per,%.2f,ns\n",
           T, R, per_atomic * 1e9);

    return 0;
}
