use rayon::prelude::*;
use std::time::Instant;
use std::env;


const PROBLEM_SIZES: [usize; 5] = [256, 512, 1024, 1536, 2048];
const THREAD_COUNTS: [usize; 5] = [1, 2, 4, 8, 16];

type Matrix = Vec<Vec<f64>>;

fn create_matrix(n: usize, init_value: f64) -> Matrix {
    vec![vec![init_value; n]; n]
}

fn matrix_multiply_sequential(a: &Matrix, b: &Matrix, n: usize) -> Matrix {
    let mut c = create_matrix(n, 0.0);
    
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    
    c
}

fn matrix_multiply_parallel(a: &Matrix, b: &Matrix, n: usize) -> Matrix {
    let mut c = create_matrix(n, 0.0);
    
    c.par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i][k] * b[k][j];
                }
                row[j] = sum;
            }
        });
    
    c
}

fn verify_results(sequential: &Matrix, parallel: &Matrix, n: usize) -> bool {
    const EPSILON: f64 = 1e-6;
    
    for i in 0..n {
        for j in 0..n {
            if (sequential[i][j] - parallel[i][j]).abs() > EPSILON {
                return false;
            }
        }
    }
    true
}

fn run_benchmark(n: usize, threads: usize) -> (f64, f64, f64) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    
    let a = create_matrix(n, 1.0);
    let b = create_matrix(n, 2.0);
    
    if n >= 256 {
        let warm_n = 128;
        let warm_a = create_matrix(warm_n, 1.0);
        let warm_b = create_matrix(warm_n, 2.0);
        let _ = matrix_multiply_parallel(&warm_a, &warm_b, warm_n);
    }
    

    let seq_time = if threads == 1 {
        let start = Instant::now();
        let _ = matrix_multiply_sequential(&a, &b, n);
        start.elapsed().as_secs_f64()
    } else {
        0.0 // Skip sequential for multi-threaded runs
    };
    
    // parallel version
    let start = Instant::now();
    let result_parallel = matrix_multiply_parallel(&a, &b, n);
    let par_time = start.elapsed().as_secs_f64();
    
    // Verify correctness (only when we have sequential result)
    if threads == 1 {
        let result_sequential = matrix_multiply_sequential(&a, &b, n);
        if !verify_results(&result_sequential, &result_parallel, n) {
            eprintln!("Warning: Results do not match for n={}, threads={}", n, threads);
        }
    }
    
    // efficiency
    let efficiency = if threads == 1 {
        1.0
    } else {
        // need T=1 time for this problem size to calculate efficiency
        // for now calculate relative efficiency
        0.0 //  calculated later with baseline
    };
    
    (seq_time, par_time, efficiency)
}

fn run_scalability_study() {
    println!("=== Rust Matrix Multiply Benchmark (Scalability) ===");
    println!("Testing problem sizes: {:?}", PROBLEM_SIZES);
    println!("Testing thread counts: {:?}", THREAD_COUNTS);
    println!();
    
    let mut baselines: Vec<f64> = Vec::new();
    
    for &n in &PROBLEM_SIZES {
        println!("\n{'=':.>60}");
        println!("Problem Size: n = {}", n);
        println!("{'=':.>60}");
        
        let mut baseline_time = 0.0;
        
        for &threads in &THREAD_COUNTS {
            print!("Threads = {:2} ... ", threads);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            
            let (seq_time, par_time, _) = run_benchmark(n, threads);
            
            if threads == 1 {
                baseline_time = par_time;
                println!("Time: {:.6}s (baseline)", par_time);
            } else {
                let speedup = baseline_time / par_time;
                let efficiency = speedup / threads as f64;
                println!("Time: {:.6}s, Speedup: {:.2}x, Efficiency: {:.2}%", 
                         par_time, speedup, efficiency * 100.0);
            }
        }
        
        baselines.push(baseline_time);
    }
    
    println!("\n\n{'=':.>60}");
    println!("Summary: Execution Times (seconds)");
    println!("{'=':.>60}");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}", 
             "n \\ T", "1", "2", "4", "8", "16");
    println!("{:-<60}", "");
    
    for &n in &PROBLEM_SIZES {
        print!("{:>8}", n);
        for &threads in &THREAD_COUNTS {
            let (_, par_time, _) = run_benchmark(n, threads);
            print!(" {:>10.4}", par_time);
        }
        println!();
    }
    
    println!("\n{'=':.>60}");
    println!("Scalability Metrics");
    println!("{'=':.>60}");
    println!("Strong Scaling: Fixed problem size, varying threads");
    println!("Efficiency = Speedup / Number of Threads");
    println!("Ideal efficiency = 100% (linear scaling)");
    println!();
    println!("Code Characteristics:");
    println!("  - Data parallelism using Rayon");
    println!("  - Automatic load balancing");
    println!("  - No explicit thread management");
    println!("  - Memory-safe concurrent access");
}

fn main() {

    let args: Vec<String> = env::args().collect();
    
    if args.len() == 3 {
        let n: usize = args[1].parse().expect("Invalid problem size");
        let threads: usize = args[2].parse().expect("Invalid thread count");
        
        println!("Running single benchmark: n={}, threads={}", n, threads);
        let (seq_time, par_time, _) = run_benchmark(n, threads);
        
        if threads == 1 {
            println!("Time: {:.6}s", par_time);
        } else {
            println!("Parallel time: {:.6}s", par_time);
        }
    } else {
        run_scalability_study();
    }
}
