// Histogram benchmark for "amount of control" (OpenMP version)
// Strategies:
//   1) G-Atomic: single shared histogram with #pragma omp atomic
//   2) TL-Local: thread-local histograms + manual reduction
//
// Usage:
//   ./hist_openmp <strategy> <dist> <N> <T> [sched] [chunk] [pad] [affinity]
//   strategy: atomic | local
//   dist:     uniform | skewed
//   N:        number of elements (e.g., 10000000)
//   T:        number of threads (e.g., 1,2,4,8,16)
//   sched:    static | dynamic | guided (default: static)
//   chunk:    chunk size (0 = runtime default)
//   pad:      0 | 1 (atomic only; 1 = padded bins)
//   affinity: 0 | 1 (0 = no pinning, 1 = pin threads to cores)
//
// Output (CSV-style):
//   hist,openmp,strategy=atomic,dist=uniform,N=10000000,T=8,sched=static,chunk=0,pad=0,affinity=0,time,0.123456,sec
//   hist,openmp,strategy=atomic,dist=uniform,N=10000000,T=8,sched=static,chunk=0,pad=0,affinity=0,correct,1,boolean

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

// For padded atomic bins (to reduce false sharing)
typedef struct {
    unsigned long long value;
    char pad[64 - sizeof(unsigned long long)]; // assume 64B cache line
} padded_bin_t;

// Strategy 1a: Global shared histogram with atomic increments (un-padded)
static double hist_atomic(const uint8_t *data,
                          unsigned long long *hist,
                          long long N,
                          int T,
                          int use_affinity) {
    for (int b = 0; b < BINS; ++b) hist[b] = 0ULL;

    omp_set_num_threads(T);

    double t0 = omp_get_wtime();
    
    // Use OpenMP's native proc_bind for thread affinity
    if (use_affinity) {
        #pragma omp parallel proc_bind(close)
        {
            #pragma omp for schedule(runtime)
            for (long long i = 0; i < N; ++i) {
                uint8_t v = data[i];
                #pragma omp atomic
                hist[v] += 1ULL;
            }
        }
    } else {
        #pragma omp parallel
        {
            #pragma omp for schedule(runtime)
            for (long long i = 0; i < N; ++i) {
                uint8_t v = data[i];
                #pragma omp atomic
                hist[v] += 1ULL;
            }
        }
    }
    
    double t1 = omp_get_wtime();
    return t1 - t0;
}

// Strategy 1b: Global shared histogram with atomic increments (padded bins)
static double hist_atomic_padded(const uint8_t *data,
                                 padded_bin_t *hist_padded,
                                 long long N,
                                 int T,
                                 int use_affinity) {
    for (int b = 0; b < BINS; ++b) hist_padded[b].value = 0ULL;

    omp_set_num_threads(T);

    double t0 = omp_get_wtime();
    
    // Use OpenMP's native proc_bind for thread affinity
    if (use_affinity) {
        #pragma omp parallel proc_bind(close)
        {
            #pragma omp for schedule(runtime)
            for (long long i = 0; i < N; ++i) {
                uint8_t v = data[i];
                #pragma omp atomic
                hist_padded[v].value += 1ULL;
            }
        }
    } else {
        #pragma omp parallel
        {
            #pragma omp for schedule(runtime)
            for (long long i = 0; i < N; ++i) {
                uint8_t v = data[i];
                #pragma omp atomic
                hist_padded[v].value += 1ULL;
            }
        }
    }
    
    double t1 = omp_get_wtime();
    return t1 - t0;
}

// Strategy 2: Thread-local histograms + manual reduction
static double hist_local(const uint8_t *data,
                         unsigned long long *hist,
                         long long N,
                         int T,
                         int use_affinity) {
    for (int b = 0; b < BINS; ++b) hist[b] = 0ULL;

    omp_set_num_threads(T);

    double t0 = omp_get_wtime();
    
    // Use OpenMP's native proc_bind for thread affinity
    if (use_affinity) {
        #pragma omp parallel proc_bind(close)
        {
            // Each thread gets its own local histogram on the stack
            unsigned long long local_hist[BINS];
            for (int b = 0; b < BINS; ++b) local_hist[b] = 0ULL;

            #pragma omp for schedule(runtime)
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
    } else {
        #pragma omp parallel
        {
            // Each thread gets its own local histogram on the stack
            unsigned long long local_hist[BINS];
            for (int b = 0; b < BINS; ++b) local_hist[b] = 0ULL;

            #pragma omp for schedule(runtime)
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
                "usage: %s <strategy> <dist> <N> <T> [sched] [chunk] [pad]\n"
                "  strategy: atomic | local\n"
                "  dist:     uniform | skewed\n"
                "  N:        number of elements (e.g. 10000000)\n"
                "  T:        threads (e.g. 1,2,4,8,16)\n"
                "  sched:    static | dynamic | guided (default: static)\n"
                "  chunk:    chunk size (0 = runtime default)\n"
                "  pad:      0 | 1 (atomic only; 1 = padded bins)\n",
                argv[0]);
        return 1;
    }

    const char *strategy = argv[1];   // "atomic" or "local"
    const char *dist     = argv[2];   // "uniform" or "skewed"
    long long N          = atoll(argv[3]);
    int T                = atoi(argv[4]);

    const char *sched = (argc > 5) ? argv[5] : "static";
    int chunk         = (argc > 6) ? atoi(argv[6]) : 0;
    int pad           = (argc > 7) ? atoi(argv[7]) : 0;
    int affinity      = (argc > 8) ? atoi(argv[8]) : 0;

    if (N <= 0 || T <= 0) {
        fprintf(stderr, "N and T must be positive.\n");
        return 1;
    }

    // Configure OpenMP schedule via runtime
    if (chunk < 0) chunk = 0;
    if (strcmp(sched, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk);
    } else if (strcmp(sched, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk);
    } else {
        // default to static
        omp_set_schedule(omp_sched_static, chunk);
        sched = "static"; // normalize
    }

    uint8_t *data = (uint8_t*) malloc((size_t)N * sizeof(uint8_t));
    unsigned long long hist[BINS];
    padded_bin_t hist_padded[BINS];

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
        if (pad) {
            elapsed = hist_atomic_padded(data, hist_padded, N, T, affinity);
            // Copy back to plain hist for checking/printing
            for (int b = 0; b < BINS; ++b) {
                hist[b] = hist_padded[b].value;
            }
        } else {
            elapsed = hist_atomic(data, hist, N, T, affinity);
        }
    } else if (strcmp(strategy, "local") == 0) {
        pad = 0; // ignore pad in local strategy
        elapsed = hist_local(data, hist, N, T, affinity);
    } else {
        fprintf(stderr, "unknown strategy: %s (use atomic|local)\n", strategy);
        free(data);
        return 1;
    }

    int correct = check_correct(hist, N);

    // CSV-style output (extended)
    printf("hist,openmp,strategy=%s,dist=%s,N=%lld,T=%d,sched=%s,chunk=%d,pad=%d,affinity=%d,time,%.6f,sec\n",
           strategy, dist, N, T, sched, chunk, pad, affinity, elapsed);
    printf("hist,openmp,strategy=%s,dist=%s,N=%lld,T=%d,sched=%s,chunk=%d,pad=%d,affinity=%d,correct,%d,boolean\n",
           strategy, dist, N, T, sched, chunk, pad, affinity, correct);

    free(data);
    return correct ? 0 : 3;
}
