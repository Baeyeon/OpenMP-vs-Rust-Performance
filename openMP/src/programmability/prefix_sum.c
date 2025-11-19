// Prefix Sum (exclusive scan) -- OpenMP version (for Programmability benchmark)
// Setup: N = 10^7, A[i]=1, T = 8; correctness only, no performance timing.
// Architecture: Two-phase block-wise scan
//   1) Each thread performs a serial exclusive scan on its own block and records its total sum in block_sum[tid]
//   2) The main thread performs a serial prefix sum on block_sum to produce block_off[]
//   3) Each thread adds its block offset back to its own section in parallel
//
// Parallel constructs used (for programmability metric counting):
//   - omp_set_num_threads           (thread configuration)
//   - #pragma omp parallel          (parallel regions, used twice)

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define N (10000000LL)  // Input array length (fixed at 1e7)
#define T (8)            // Number of threads (fixed at 8)

int main(void) {

    omp_set_num_threads(T);

    long long *in  = (long long*) malloc(sizeof(long long) * N);
    long long *out = (long long*) malloc(sizeof(long long) * N);
    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 2;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long beg = (N * tid) / T;
        long long end = (N * (tid + 1)) / T;
        for (long long i = beg; i < end; ++i) {
            in[i] = 1;
        }
    }

    // arrays for per-block sums and offsets
    long long *block_sum = (long long*) malloc(sizeof(long long) * T);
    long long *block_off = (long long*) malloc(sizeof(long long) * T);
    if (!block_sum || !block_off) {
        fprintf(stderr, "Memory allocation failed\n");
        free(in); free(out);
        return 2;
    }

    // Phase 1: Each thread performs an exclusive scan within its own block
    //           and records the total block sum into block_sum[tid]
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long beg = (N * tid) / T;
        long long end = (N * (tid + 1)) / T;

        long long run = 0; 
        for (long long i = beg; i < end; ++i) {
            out[i] = run;     // Exclusive scan: write prefix before adding current element
            run   += in[i];
        }
        block_sum[tid] = run;  
    }

    // Phase 1.5: Serial prefix sum over block_sum[] to compute each block’s global offset
    // (Since T ≪ N, serial computation here is negligible. Could be parallelized if T is large.)
    long long acc = 0;
    for (int t = 0; t < T; ++t) {
        block_off[t] = acc;
        acc += block_sum[t];
    }

    // Phase 2: Add each block’s offset to its section of the output array in parallel
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long beg = (N * tid) / T;
        long long end = (N * (tid + 1)) / T;
        long long off = block_off[tid];

        for (long long i = beg; i < end; ++i) {
            out[i] += off;
        }
    }


    int ok = 1;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long beg = (N * tid) / T;
        long long end = (N * (tid + 1)) / T;

        int local_ok = 1;
        for (long long i = beg; i < end; ++i) {
            if (out[i] != i) { local_ok = 0; break; }
        }
        #pragma omp critical
        {
            if (!local_ok) ok = 0;
        }
    }

    printf("bench=scan lang=openmp N=%lld T=%d correct=%d\n", (long long)N, T, ok);

    free(block_off); free(block_sum);
    free(out); free(in);
    return ok ? 0 : 3;
}
