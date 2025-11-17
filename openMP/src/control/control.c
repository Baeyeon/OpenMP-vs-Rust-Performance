// Histogram benchmark for "amount of control" (OpenMP version)
// Strategies:
//   1) G-Atomic: single shared histogram with #pragma omp atomic
//   2) TL-Local: thread-local histograms + manual reduction
//
// Usage:
//   ./hist_openmp <strategy> <dist> <N> <T>
//   strategy: "atomic" or "local"
//   dist:     "uniform" or "skewed"
//   N:        number of elements (e.g., 10000000)
//   T:        number of threads (e.g., 1,2,4,8,16)
//
// Output (CSV-style):
//   hist,openmp,strategy=atomic,dist=uniform,N=10000000,T=8,time,0.123456,sec
//   hist,openmp,strategy=atomic,dist=uniform,N=10000000,T=8,correct,1,boolean

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define BINS 256

// Simple LCG RNG (deterministic)
static inline uint32_t lcg_next(uint32_t x) {
    return x * 1664525u + 1013904223u;
}

// Generate uniform [0,255]
static void gen_uniform(uint8_t *data, long long N) {
    uint32_t x = 123456789u;
    for (long long i = 0; i < N; ++i) {
        x = lcg_next(x);
        data[i] = (uint8_t)(x & 0xFF);  // use low 8 bits
    }
}

// Generate skewed distribution: ~80% in first 20% bins (0..51)
static void gen_skewed(uint8_t *data, long long N) {
    const int hot_bins = (int)(BINS * 0.2); // 51
    uint32_t x = 987654321u;
    const uint32_t threshold = (uint32_t)(0.8 * 4294967295.0); // ~80%

    for (long long i = 0; i < N; ++i) {
        x = lcg_next(x);
        if (x < threshold) {
            // hot range
            data[i] = (uint8_t)(x % hot_bins); // 0 .. hot_bins-1
        } else {
            // cold range
            uint8_t v = (uint8_t)(x & 0xFF);
            if (v < hot_bins) v += hot_bins;
            data[i] = v;
        }
    }
}

// Strategy 1: Global shared histogram with atomic increments
static double hist_atomic(const uint8_t *data,
                          unsigned long long *hist,
                          long long N,
                          int T) {
    for (int b = 0; b < BINS; ++b) hist[b] = 0;

    omp_set_num_threads(T);

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long long i = 0; i < N; ++i) {
            uint8_t v = data[i];
            #pragma omp atomic
            hist[v] += 1ULL;
        }
    }
    double t1 = omp_get_wtime();
    return t1 - t0;
}

// Strategy 2: Thread-local histograms + manual reduction
static double hist_local(const uint8_t *data,
                         unsigned long long *hist,
                         long long N,
                         int T) {
    for (int b = 0; b < BINS; ++b) hist[b] = 0;

    omp_set_num_threads(T);

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        // Each thread gets its own local histogram on the stack
        unsigned long long local_hist[BINS];
        for (int b = 0; b < BINS; ++b) local_hist[b] = 0ULL;

        #pragma omp for schedule(static)
        for (long long i = 0; i < N; ++i) {
            uint8_t v = data[i];
            local_hist[v] += 1ULL;
        }

        // Merge local_hist into global hist
        #pragma omp critical
        {
            for (int b = 0; b < BINS; ++b) {
                hist[b] += local_hist[b];
            }
        }
    }
    double t1 = omp_get_wtime();
    return t1 - t0;
}

// Check that sum(hist) == N
static int check_correct(const unsigned long long *hist, long long N) {
    unsigned long long total = 0;
    for (int b = 0; b < BINS; ++b) {
        total += hist[b];
    }
    return (total == (unsigned long long)N) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <strategy> <dist> <N> <T>\n"
                "  strategy: atomic | local\n"
                "  dist:     uniform | skewed\n"
                "  N:        number of elements (e.g. 10000000)\n"
                "  T:        threads (e.g. 1,2,4,8,16)\n",
                argv[0]);
        return 1;
    }

    const char *strategy = argv[1];   // "atomic" or "local"
    const char *dist     = argv[2];   // "uniform" or "skewed"
    long long N          = atoll(argv[3]);
    int T                = atoi(argv[4]);

    if (N <= 0 || T <= 0) {
        fprintf(stderr, "N and T must be positive.\n");
        return 1;
    }

    uint8_t *data = (uint8_t*) malloc((size_t)N * sizeof(uint8_t));
    unsigned long long hist[BINS];

    if (!data) {
        fprintf(stderr, "malloc failed for data\n");
        return 2;
    }

    // Generate input data (not timed)
    if (strcmp(dist, "uniform") == 0) {
        gen_uniform(data, N);
    } else if (strcmp(dist, "skewed") == 0) {
        gen_skewed(data, N);
    } else {
        fprintf(stderr, "unknown dist: %s (use uniform|skewed)\n", dist);
        free(data);
        return 1;
    }

    // Run the chosen strategy
    double elapsed = 0.0;
    if (strcmp(strategy, "atomic") == 0) {
        elapsed = hist_atomic(data, hist, N, T);
    } else if (strcmp(strategy, "local") == 0) {
        elapsed = hist_local(data, hist, N, T);
    } else {
        fprintf(stderr, "unknown strategy: %s (use atomic|local)\n", strategy);
        free(data);
        return 1;
    }

    int correct = check_correct(hist, N);

    // CSV-style output
    printf("hist,openmp,strategy=%s,dist=%s,N=%lld,T=%d,time,%.6f,sec\n",
           strategy, dist, N, T, elapsed);
    printf("hist,openmp,strategy=%s,dist=%s,N=%lld,T=%d,correct,%d,boolean\n",
           strategy, dist, N, T, correct);

    free(data);
    return correct ? 0 : 3;
}
