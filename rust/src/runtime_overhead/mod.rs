// Runtime Overhead Benchmarks
// Measures the cost of thread operations and synchronization primitives

use rayon;
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];
const ITERATIONS: &[usize] = &[10_000, 25_000, 50_000, 75_000, 100_000];

pub fn run_all_benchmarks() {
    // CSV output format matching OpenMP for easy comparison and data processing
    spawn_join_benchmark();
    barrier_benchmark();
    mutex_benchmark();
    atomic_benchmark();
}

/// 1: Parallel Scope (Rayon)
/// overhead of parallel regions using Rayon thread pool (comparable to OpenMP)
/// Measures cost per parallel scope creation (like OpenMP's parallel region)
fn spawn_join_benchmark() {
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            // Create thread pool with specific size (like OpenMP's omp_set_num_threads)
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();
            
            let start = Instant::now();
            
            pool.install(|| {
                for _ in 0..iterations {
                    // Use rayon::scope - creates parallel region like OpenMP
                    rayon::scope(|s| {
                        for _ in 0..num_threads {
                            s.spawn(|_| {
                                // Empty body - purely measure parallel region overhead
                            });
                        }
                    });
                }
            });
            
            let duration = start.elapsed();
            // cost per parallel scope,
            let total_ms = duration.as_secs_f64() * 1000.0;
            let avg_ns = duration.as_nanos() as f64 / iterations as f64;
            
            println!("overhead,rust,T={},R={},parallel_total,{:.6},ms",
                num_threads, iterations, total_ms);
            println!("overhead,rust,T={},R={},parallel_per,{:.3},ns",
                num_threads, iterations, avg_ns);
        }
    }
}

/// 2: Barrier Synchronization
/// overhead of barrier synchronization using Rayon thread pool
fn barrier_benchmark() {
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();
            
            let barrier = Arc::new(Barrier::new(num_threads));
            let start = Instant::now();
            
            pool.install(|| {
                rayon::scope(|s| {
                    for _ in 0..num_threads {
                        let barrier_clone = Arc::clone(&barrier);
                        s.spawn(move |_| {
                            for _ in 0..iterations {
                                barrier_clone.wait();
                            }
                        });
                    }
                });
            });
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let total_ms = duration.as_secs_f64() * 1000.0;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("overhead,rust,T={},R={},barrier_total,{:.6},ms",
                num_threads, iterations, total_ms);
            println!("overhead,rust,T={},R={},barrier_per,{:.3},ns",
                num_threads, iterations, avg_ns);
        }
    }
}

/// 3: Mutex Lock/Unlock
/// overhead of mutex operations using Rayon thread pool
fn mutex_benchmark() {
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();
            
            let counter = Arc::new(Mutex::new(0u64));
            let start = Instant::now();
            
            pool.install(|| {
                rayon::scope(|s| {
                    for _ in 0..num_threads {
                        let counter_clone = Arc::clone(&counter);
                        s.spawn(move |_| {
                            for _ in 0..iterations {
                                let mut val = counter_clone.lock().unwrap();
                                *val += 1;
                                // lock is automatically released here
                            }
                        });
                    }
                });
            });
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let total_ms = duration.as_secs_f64() * 1000.0;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("overhead,rust,T={},R={},mutex_total,{:.6},ms",
                num_threads, iterations, total_ms);
            println!("overhead,rust,T={},R={},mutex_per,{:.3},ns",
                num_threads, iterations, avg_ns);
        }
    }
}

/// 4: Atomic Operations
/// overhead of atomic fetch_add operations using Rayon thread pool
fn atomic_benchmark() {
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap();
            
            let counter = Arc::new(AtomicU64::new(0));
            let start = Instant::now();
            
            pool.install(|| {
                rayon::scope(|s| {
                    for _ in 0..num_threads {
                        let counter_clone = Arc::clone(&counter);
                        s.spawn(move |_| {
                            for _ in 0..iterations {
                                counter_clone.fetch_add(1, Ordering::SeqCst);
                            }
                        });
                    }
                });
            });
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let total_ms = duration.as_secs_f64() * 1000.0;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("overhead,rust,T={},R={},atomic_total,{:.6},ms",
                num_threads, iterations, total_ms);
            println!("overhead,rust,T={},R={},atomic_per,{:.3},ns",
                num_threads, iterations, avg_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_spawn_join() {
        // Simple smoke test
        let handles: Vec<_> = (0..4)
            .map(|_| thread::spawn(|| 1 + 1))
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
    
    #[test]
    fn test_barrier() {
        let barrier = Arc::new(Barrier::new(2));
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
        });
        
        barrier.wait();
        handle.join().unwrap();
    }
    
    #[test]
    fn test_mutex() {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = Arc::clone(&counter);
        
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let mut val = counter_clone.lock().unwrap();
                *val += 1;
            }
        });
        
        handle.join().unwrap();
        assert_eq!(*counter.lock().unwrap(), 100);
    }
    
    #[test]
    fn test_atomic() {
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = Arc::clone(&counter);
        
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        
        handle.join().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }
}
