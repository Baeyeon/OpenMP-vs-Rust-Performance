// Runtime Overhead benchmark (OpenMP)
// Measures:
//   1) Parallel region overhead: repeated `#pragma omp parallel {}`
//   2) Barrier overhead: single parallel region with repeated `#pragma omp barrier`
//
// Usage:
//   ./overhead_openmp <T> <R>
//   T = number of threads (e.g., 1,2,4,8,16)
//   R = number of repetitions (e.g., 100000)
//
// Output: CSV-style lines, e.g.:
//   overhead,openmp,T=8,R=100000,parallel_total,0.012345,sec
//   overhead,openmp,T=8,R=100000,parallel_per,1.234e-07,sec
//   overhead,openmp,T=8,R=100000,barrier_total,0.034567,sec
//   overhead,openmp,T=8,R=100000,barrier_per,3.456e-07,sec

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
    double per_barrier  = time_barrier / (double)R;

    // CSV-style output, easy to parse in Python/R
    printf("overhead,openmp,T=%d,R=%lld,parallel_total,%.9f,sec\n",
           T, R, time_parallel);
    printf("overhead,openmp,T=%d,R=%lld,parallel_per,%.9e,sec\n",
           T, R, per_parallel);
    printf("overhead,openmp,T=%d,R=%lld,barrier_total,%.9f,sec\n",
           T, R, time_barrier);
    printf("overhead,openmp,T=%d,R=%lld,barrier_per,%.9e,sec\n",
           T, R, per_barrier);

    return 0;
}
