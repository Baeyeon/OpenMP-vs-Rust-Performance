use rayon::prelude::*;
use std::time::Instant;

const N: usize = 10_000_000; // 10^7
const THREADS: usize = 8;
const INPUT_VALUE: u64 = 1;

fn prefix_sum_sequential(arr: &[u64]) -> Vec<u64> {
    let mut result = vec![0u64; arr.len()];
    result[0] = arr[0];
    for i in 1..arr.len() {
        result[i] = result[i - 1] + arr[i];
    }
    result
}

fn prefix_sum_parallel(arr: &[u64]) -> Vec<u64> {
    // parallel prefix sum
    let n = arr.len();
    let chunk_size = (n + THREADS - 1) / THREADS;
    
    // local prefix sums in parallel
    let local_sums: Vec<Vec<u64>> = arr
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local = vec![0u64; chunk.len()];
            local[0] = chunk[0];
            for i in 1..chunk.len() {
                local[i] = local[i - 1] + chunk[i];
            }
            local
        })
        .collect();
    
    // compute offsets from last element of each chunk
    let mut offsets = vec![0u64; local_sums.len()];
    for i in 1..offsets.len() {
        offsets[i] = offsets[i - 1] + local_sums[i - 1].last().unwrap();
    }

    // add offsets to local sums in parallel
    let result: Vec<u64> = local_sums
        .into_par_iter()
        .zip(offsets.into_par_iter())
        .flat_map_iter(|(local, offset)| {
            local.into_iter().map(move |val| val + offset)
        })
        .collect();
    
    result
}

fn verify_results(sequential: &[u64], parallel: &[u64]) -> bool {
    if sequential.len() != parallel.len() {
        return false;
    }
    sequential.iter().zip(parallel.iter()).all(|(s, p)| s == p)
}

fn main() {
    // thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build_global()
        .unwrap();
    
    println!("=== Rust Prefix Sum Benchmark (Programmability) ===");
    println!("Array size: N = {}", N);
    println!("Threads: T = {}", THREADS);
    println!("Input value: {}", INPUT_VALUE);
    println!();
    
    // Init input array
    let input: Vec<u64> = vec![INPUT_VALUE; N];
    
    // warm-up 
    let _ = prefix_sum_parallel(&input[..1000]);
    
    // sequential
    println!("Running sequential version...");
    let start = Instant::now();
    let sequential_result = prefix_sum_sequential(&input);
    let seq_time = start.elapsed();
    println!("Sequential time: {:.6} seconds", seq_time.as_secs_f64());
    
    // parallel 
    println!("Running parallel version...");
    let start = Instant::now();
    let parallel_result = prefix_sum_parallel(&input);
    let par_time = start.elapsed();
    println!("Parallel time: {:.6} seconds", par_time.as_secs_f64());
    
    //correctness
    println!("\nVerifying results...");
    if verify_results(&sequential_result, &parallel_result) {
        println!("✓ Results match!");
    } else {
        println!("✗ Results do not match!");
        return;
    }
    
    //  speedup
    let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);
    
}
