// Histogram Benchmark - Controllability Comparison
// Tests programmer control over thread-to-core assignment and variable sharing/privacy

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;
use rand::Rng;

const PROBLEM_SIZES: &[usize] = &[10_000_000, 100_000_000]; // 10M, 100M
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];
const NUM_BINS: usize = 256;

fn main() {
    println!("Histogram Controllability Benchmark");
    println!("====================================\n");
    println!("Testing programmer control over:");
    println!("  1. Variable sharing (G-Atomic: shared atomic counters)");
    println!("  2. Variable privacy (TL-Local: thread-local histograms)");
    println!("  3. Thread affinity (core pinning)\n");
    
    // Generate test data
    println!("Generating test data...");
    let data_10m = generate_data(PROBLEM_SIZES[0]);
    let data_100m = generate_data(PROBLEM_SIZES[1]);
    println!("Data generation complete.\n");
    
    for &size in PROBLEM_SIZES {
        let data = if size == PROBLEM_SIZES[0] { &data_10m } else { &data_100m };
        
        println!("===========================================");
        println!("Problem Size: {} elements ({:.1}M)", size, size as f64 / 1_000_000.0);
        println!("===========================================\n");
        
        run_strategy_1_g_atomic(data, size);
        println!();
        
        run_strategy_2_tl_local(data, size);
        println!();
        
        run_strategy_3_thread_affinity(data, size);
        println!();
    }
    
    print_controllability_summary();
}

/// random test data 
fn generate_data(size: usize) -> Arc<Vec<u8>> {
    let mut rng = rand::thread_rng();
    Arc::new((0..size).map(|_| rng.gen::<u8>()).collect())
}

/// 1: G-Atomic (Global Atomic Counters)
/// Explicit control over shared variables using atomic operations
fn run_strategy_1_g_atomic(data: &Arc<Vec<u8>>, size: usize) {
    println!("Strategy 1: G-Atomic (Global Atomic Counters)");
    println!("----------------------------------------------");
    println!("Control Feature: Explicit shared variable with atomic operations");
    println!();
    println!("Threads | Time (ms) | Throughput (M elem/s) | Speedup | Efficiency");
    println!("--------|-----------|----------------------|---------|------------");
    
    let mut baseline_time = 0.0;
    
    for &num_threads in THREAD_COUNTS {
        // Create shared atomic histogram - EXPLICIT CONTROL
        let histogram: Arc<Vec<AtomicU64>> = Arc::new(
            (0..NUM_BINS).map(|_| AtomicU64::new(0)).collect()
        );
        
        let start = Instant::now();
        
        if num_threads == 1 {
            // Single-threaded baseline
            let data_clone = Arc::clone(data);
            for &val in data_clone.iter() {
                histogram[val as usize].fetch_add(1, Ordering::Relaxed);
            }
        } else {
            // Multi-threaded
            let chunk_size = (size + num_threads - 1) / num_threads;
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let data_clone = Arc::clone(data);
                    let hist_clone = Arc::clone(&histogram);
                    
                    thread::spawn(move || {
                        let start_idx = tid * chunk_size;
                        let end_idx = ((tid + 1) * chunk_size).min(size);
                        
                        for i in start_idx..end_idx {
                            let val = data_clone[i] as usize;
                            hist_clone[val].fetch_add(1, Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        }
        
        let duration = start.elapsed();
        let time_ms = duration.as_secs_f64() * 1000.0;
        let throughput = (size as f64 / 1_000_000.0) / duration.as_secs_f64();
        
        if num_threads == 1 {
            baseline_time = time_ms;
        }
        
        let speedup = baseline_time / time_ms;
        let efficiency = (speedup / num_threads as f64) * 100.0;
        
        println!("{:7} | {:9.2} | {:20.2} | {:7.2} | {:9.1}%",
            num_threads, time_ms, throughput, speedup, efficiency);
        
        // Verify correctness
        let total: u64 = histogram.iter().map(|x| x.load(Ordering::Relaxed)).sum();
        assert_eq!(total as usize, size, "Histogram count mismatch!");
    }
    
}

/// 2: TL-Local (Thread-Local Private Histograms)
/// Control over variable privacy through ownership and manual reduction
fn run_strategy_2_tl_local(data: &Arc<Vec<u8>>, size: usize) {
    println!("Strategy 2: TL-Local (Thread-Local Private Histograms)");
    println!("-------------------------------------------------------");
    println!("Control Feature: Private variables per thread + manual reduction");
    println!();
    println!("Threads | Time (ms) | Throughput (M elem/s) | Speedup | Efficiency");
    println!("--------|-----------|----------------------|---------|------------");
    
    let mut baseline_time = 0.0;
    
    for &num_threads in THREAD_COUNTS {
        let start = Instant::now();
        
        if num_threads == 1 {
            // Single-threaded baseline
            let mut histogram = [0u64; NUM_BINS];
            for &val in data.iter() {
                histogram[val as usize] += 1;
            }
            
            let duration = start.elapsed();
            let time_ms = duration.as_secs_f64() * 1000.0;
            baseline_time = time_ms;
            let throughput = (size as f64 / 1_000_000.0) / duration.as_secs_f64();
            
            println!("{:7} | {:9.2} | {:20.2} | {:7.2} | {:9.1}%",
                1, time_ms, throughput, 1.0, 100.0);
            
            let total: u64 = histogram.iter().sum();
            assert_eq!(total as usize, size);
        } else {
            // Multi-threaded with private histograms - EXPLICIT PRIVACY CONTROL
            let chunk_size = (size + num_threads - 1) / num_threads;
            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let data_clone = Arc::clone(data);
                    
                    thread::spawn(move || {
                        // Private histogram - automatic through move semantics
                        let mut local_hist = [0u64; NUM_BINS];
                        
                        let start_idx = tid * chunk_size;
                        let end_idx = ((tid + 1) * chunk_size).min(size);
                        
                        for i in start_idx..end_idx {
                            local_hist[data_clone[i] as usize] += 1;
                        }
                        
                        local_hist
                    })
                })
                .collect();
            
            // Manual reduction - EXPLICIT MERGE CONTROL
            let mut global_histogram = [0u64; NUM_BINS];
            for handle in handles {
                let local_hist = handle.join().unwrap();
                for i in 0..NUM_BINS {
                    global_histogram[i] += local_hist[i];
                }
            }
            
            let duration = start.elapsed();
            let time_ms = duration.as_secs_f64() * 1000.0;
            let throughput = (size as f64 / 1_000_000.0) / duration.as_secs_f64();
            let speedup = baseline_time / time_ms;
            let efficiency = (speedup / num_threads as f64) * 100.0;
            
            println!("{:7} | {:9.2} | {:20.2} | {:7.2} | {:9.1}%",
                num_threads, time_ms, throughput, speedup, efficiency);
            
            let total: u64 = global_histogram.iter().sum();
            assert_eq!(total as usize, size);
        }
    }
    
    println!("\nControllability: Rust provides implicit privacy via ownership, requires manual reduction");
}

/// 3: Thread Affinity Control
/// Control over thread-to-core assignment
fn run_strategy_3_thread_affinity(data: &Arc<Vec<u8>>, size: usize) {
    println!("Strategy 3: Thread Affinity Control");
    println!("------------------------------------");
    println!("Control Feature: Pin threads to specific CPU cores");
    println!();
    
    // if core_affinity is available
    let core_ids = core_affinity::get_core_ids();
    if core_ids.is_none() {
        println!("Core affinity not supported on this system.");
        println!("Controllability: Requires external 'core_affinity' crate and platform support\n");
        return;
    }
    
    let core_ids = core_ids.unwrap();
    println!("Available CPU cores: {}", core_ids.len());
    println!();
    println!("Threads | Time (ms) | Throughput (M elem/s) | Speedup | Efficiency");
    println!("--------|-----------|----------------------|---------|------------");
    
    let mut baseline_time = 0.0;
    
    for &num_threads in THREAD_COUNTS {
        if num_threads > core_ids.len() {
            println!("{:7} | Skipped (exceeds available cores)", num_threads);
            continue;
        }
        
        let histogram: Arc<Vec<AtomicU64>> = Arc::new(
            (0..NUM_BINS).map(|_| AtomicU64::new(0)).collect()
        );
        
        let start = Instant::now();
        
        // Always spawn threads (even for single thread) to avoid pinning main thread
        let chunk_size = (size + num_threads - 1) / num_threads;
        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let data_clone = Arc::clone(data);
                let hist_clone = Arc::clone(&histogram);
                let core_id = core_ids[tid];
                
                thread::spawn(move || {
                    // EXPLICIT CORE PINNING CONTROL
                    core_affinity::set_for_current(core_id);
                    
                    let start_idx = tid * chunk_size;
                    let end_idx = ((tid + 1) * chunk_size).min(size);
                    
                    for i in start_idx..end_idx {
                        let val = data_clone[i] as usize;
                        hist_clone[val].fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let duration = start.elapsed();
        let time_ms = duration.as_secs_f64() * 1000.0;
        let throughput = (size as f64 / 1_000_000.0) / duration.as_secs_f64();
        
        if num_threads == 1 {
            baseline_time = time_ms;
        }
        
        let speedup = baseline_time / time_ms;
        let efficiency = (speedup / num_threads as f64) * 100.0;
        
        println!("{:7} | {:9.2} | {:20.2} | {:7.2} | {:9.1}%",
            num_threads, time_ms, throughput, speedup, efficiency);
        
        let total: u64 = histogram.iter().map(|x| x.load(Ordering::Relaxed)).sum();
        assert_eq!(total as usize, size);
    }
    
}

fn print_controllability_summary() {

}
