// Histogram benchmark for "amount of control" (Rust/Rayon version)
// Strategies:
//   1) Rayon-Atomic: single shared histogram with atomic operations
//   2) Rayon-Local: thread-local histograms + automatic reduction
//
// Usage:
//   ./histogram <strategy> <dist> <N> <T> [grain] [pad] [affinity]
//   strategy: atomic | local
//   dist:     uniform | skewed
//   N:        number of elements (e.g., 10000000)
//   T:        number of threads (e.g., 1,2,4,8,16)
//   grain:    chunk size per task (0 = auto)
//   pad:      0 | 1 (atomic only; 1 = padded bins)
//   affinity: 0 | 1 (0 = no pinning, 1 = pin threads to cores)
//
// Output (CSV-style):
//   hist,rayon,strategy=atomic,dist=uniform,N=10000000,T=8,grain=0,pad=0,affinity=0,time,0.123456,sec
//   hist,rayon,strategy=atomic,dist=uniform,N=10000000,T=8,grain=0,pad=0,affinity=0,correct,1,boolean

use rayon::prelude::*;
use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const BINS: usize = 256;

#[repr(align(64))]
struct PaddedAtomicU64(AtomicU64);

// Global counter for thread ID assignment when using affinity
static THREAD_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Thread affinity: Pin thread to specific core
fn set_thread_affinity() -> usize {
    let thread_id = THREAD_COUNTER.fetch_add(1, Ordering::SeqCst);
    let core_ids_result = core_affinity::get_core_ids();
    
    if let Some(core_ids) = core_ids_result {
        if thread_id < core_ids.len() {
            core_affinity::set_for_current(core_ids[thread_id]);
        }
    }
    
    thread_id
}

// Simple LCG RNG (deterministic, matching OpenMP)
fn lcg_next(x: u32) -> u32 {
    x.wrapping_mul(1664525u32).wrapping_add(1013904223u32)
}

// Generate uniform distribution [0,255]
fn gen_uniform(n: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(n);
    let mut x = 123456789u32;
    for _ in 0..n {
        x = lcg_next(x);
        data.push((x & 0xFF) as u8);
    }
    data
}

// Generate skewed distribution: ~80% in first 20% bins (0..51)
fn gen_skewed(n: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(n);
    let hot_bins = (BINS as f64 * 0.2) as u8; // 51
    let threshold = (0.8 * u32::MAX as f64) as u32; // ~80%
    let mut x = 987654321u32;

    for _ in 0..n {
        x = lcg_next(x);
        let val = if x < threshold {
            // hot range
            (x % hot_bins as u32) as u8
        } else {
            // cold range
            let mut v = (x & 0xFF) as u8;
            if v < hot_bins {
                v += hot_bins;
            }
            v
        };
        data.push(val);
    }
    data
}

// Strategy 1: Rayon Atomic (Shared Histogram)
// Adds:
//   - grain: chunk size (0 = auto)
//   - pad:   if true, use cache-line padded bins
//   - use_affinity: if true, pin threads to cores
fn hist_atomic(data: &[u8], num_threads: usize, grain: usize, pad: bool, use_affinity: bool) -> (f64, Vec<u64>) {
    // Reset counter for affinity
    if use_affinity {
        THREAD_COUNTER.store(0, Ordering::SeqCst);
    }
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            if use_affinity {
                set_thread_affinity();
            }
        })
        .build()
        .unwrap();

    let start = Instant::now();

    let result: Vec<u64> = if pad {
        // Padded atomic bins to reduce false sharing
        let histogram: Vec<PaddedAtomicU64> = (0..BINS)
            .map(|_| PaddedAtomicU64(AtomicU64::new(0)))
            .collect();

        pool.install(|| {
            if grain > 0 {
                data.par_chunks(grain).for_each(|chunk| {
                    for &val in chunk {
                        histogram[val as usize]
                            .0
                            .fetch_add(1, Ordering::Relaxed);
                    }
                });
            } else {
                data.par_iter().for_each(|&val| {
                    histogram[val as usize]
                        .0
                        .fetch_add(1, Ordering::Relaxed);
                });
            }
        });

        histogram
            .iter()
            .map(|x| x.0.load(Ordering::Relaxed))
            .collect()
    } else {
        // Original contiguous atomic bins
        let histogram: Vec<AtomicU64> = (0..BINS)
            .map(|_| AtomicU64::new(0))
            .collect();

        pool.install(|| {
            if grain > 0 {
                data.par_chunks(grain).for_each(|chunk| {
                    for &val in chunk {
                        histogram[val as usize].fetch_add(1, Ordering::Relaxed);
                    }
                });
            } else {
                data.par_iter().for_each(|&val| {
                    histogram[val as usize].fetch_add(1, Ordering::Relaxed);
                });
            }
        });

        histogram
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect()
    };

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, result)
}

// Strategy 2: Rayon Local (Thread-Local Histograms)
// Adds:
//   - grain: chunk size (0 = auto: roughly N / T)
//   - use_affinity: if true, pin threads to cores
fn hist_local(data: &[u8], num_threads: usize, grain: usize, use_affinity: bool) -> (f64, Vec<u64>) {
    // Reset counter for affinity
    if use_affinity {
        THREAD_COUNTER.store(0, Ordering::SeqCst);
    }
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            if use_affinity {
                set_thread_affinity();
            }
        })
        .build()
        .unwrap();

    let start = Instant::now();

    let histogram = pool.install(|| {
        let par = if grain > 0 {
            data.par_chunks(grain)
        } else {
            let chunk_size = (data.len() + num_threads - 1) / num_threads;
            data.par_chunks(chunk_size)
        };

        par.map(|chunk| {
                let mut local_hist = [0u64; BINS];
                for &val in chunk {
                    local_hist[val as usize] += 1;
                }
                local_hist
            })
            .reduce(
                || [0u64; BINS],
                |mut acc, local| {
                    for i in 0..BINS {
                        acc[i] += local[i];
                    }
                    acc
                },
            )
    });

    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, histogram.to_vec())
}

// Check that sum(hist) == N
fn check_correct(hist: &[u64], n: usize) -> bool {
    let total: u64 = hist.iter().sum();
    total as usize == n
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "usage: {} <strategy> <dist> <N> <T> [grain] [pad]",
            args[0]
        );
        eprintln!("  strategy: atomic | local");
        eprintln!("  dist:     uniform | skewed");
        eprintln!("  N:        number of elements (e.g. 10000000)");
        eprintln!("  T:        threads (e.g. 1,2,4,8,16)");
        eprintln!("  grain:    chunk size per task (0 = auto)");
        eprintln!("  pad:      0 | 1 (atomic only; default 0)");
        std::process::exit(1);
    }

    let strategy = &args[1];
    let dist = &args[2];
    let n: usize = args[3].parse().expect("N must be a positive integer");
    let t: usize = args[4].parse().expect("T must be a positive integer");
    let grain: usize = if args.len() > 5 {
        args[5].parse().unwrap_or(0)
    } else {
        0
    };
    let pad_flag_raw: i32 = if args.len() > 6 {
        args[6].parse().unwrap_or(0)
    } else {
        0
    };
    let pad = pad_flag_raw != 0;
    let affinity_raw: i32 = if args.len() > 7 {
        args[7].parse().unwrap_or(0)
    } else {
        0
    };
    let affinity = affinity_raw != 0;

    if n == 0 || t == 0 {
        eprintln!("N and T must be positive.");
        std::process::exit(1);
    }

    // Generate input data (not timed)
    let data = match dist.as_str() {
        "uniform" => gen_uniform(n),
        "skewed" => gen_skewed(n),
        _ => {
            eprintln!("unknown dist: {} (use uniform|skewed)", dist);
            std::process::exit(1);
        }
    };

    // Run the chosen strategy
    let (elapsed, histogram) = match strategy.as_str() {
        "atomic" => hist_atomic(&data, t, grain, pad, affinity),
        "local" => hist_local(&data, t, grain, affinity),
        _ => {
            eprintln!("unknown strategy: {} (use atomic|local)", strategy);
            std::process::exit(1);
        }
    };

    let correct = check_correct(&histogram, n);
    let pad_flag = if strategy == "atomic" && pad { 1 } else { 0 };
    let affinity_flag = if affinity { 1 } else { 0 };

    // CSV-style output (extended)
    println!(
        "hist,rayon,strategy={},dist={},N={},T={},grain={},pad={},affinity={},time,{:.6},sec",
        strategy,
        dist,
        n,
        t,
        grain,
        pad_flag,
        affinity_flag,
        elapsed
    );
    println!(
        "hist,rayon,strategy={},dist={},N={},T={},grain={},pad={},affinity={},correct,{},boolean",
        strategy,
        dist,
        n,
        t,
        grain,
        pad_flag,
        affinity_flag,
        if correct { 1 } else { 0 }
    );

    if !correct {
        std::process::exit(3);
    }
}
