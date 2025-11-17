// Runtime Overhead Benchmarks
// Measures the cost of thread operations and synchronization primitives

use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];
const ITERATIONS: &[usize] = &[10_000, 25_000, 50_000, 75_000, 100_000];

pub fn run_all_benchmarks() {
    println!("Runtime Overhead Benchmarks");
    println!("===========================\n");
    
    spawn_join_benchmark();
    println!();
    barrier_benchmark();
    println!();
    mutex_benchmark();
    println!();
    atomic_benchmark();
}

/// 1: Thread Spawn + Join
/// overhead of creating and joining threads
fn spawn_join_benchmark() {
    println!("1. Spawn + Join Benchmark");
    println!("   Measures thread creation and termination overhead");
    println!("   ------------------------------------------------");
    println!("   Threads | Iterations | Total Time (ms) | Avg Cost per Op (ns)");
    println!("   --------|------------|-----------------|---------------------");
    
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let start = Instant::now();
            
            for _ in 0..iterations {
                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        thread::spawn(|| {
                            // Minimal work to isolate spawn/join overhead
                            let _ = 1 + 1;
                        })
                    })
                    .collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
            }
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("   {:7} | {:10} | {:15.2} | {:20.2}",
                num_threads, iterations, duration.as_secs_f64() * 1000.0, avg_ns);
        }
    }
}

/// 2: Barrier Synchronization
/// overhead of barrier synchronization
fn barrier_benchmark() {
    println!("2. Barrier Synchronization Benchmark");
    println!("   Measures barrier wait overhead");
    println!("   ------------------------------------------------");
    println!("   Threads | Iterations | Total Time (ms) | Avg Cost per Op (ns)");
    println!("   --------|------------|-----------------|---------------------");
    
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let barrier = Arc::new(Barrier::new(num_threads));
            let start = Instant::now();
            
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let barrier_clone = Arc::clone(&barrier);
                    thread::spawn(move || {
                        for _ in 0..iterations {
                            barrier_clone.wait();
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("   {:7} | {:10} | {:15.2} | {:20.2}",
                num_threads, iterations, duration.as_secs_f64() * 1000.0, avg_ns);
        }
    }
}

/// 3: Mutex Lock/Unlock
/// overhead of mutex operations
fn mutex_benchmark() {
    println!("3. Mutex Lock/Unlock Benchmark");
    println!("   Measures mutex contention overhead");
    println!("   ------------------------------------------------");
    println!("   Threads | Iterations | Total Time (ms) | Avg Cost per Op (ns)");
    println!("   --------|------------|-----------------|---------------------");
    
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let counter = Arc::new(Mutex::new(0u64));
            let start = Instant::now();
            
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let counter_clone = Arc::clone(&counter);
                    thread::spawn(move || {
                        for _ in 0..iterations {
                            let mut val = counter_clone.lock().unwrap();
                            *val += 1;
                            // lock is automatically released here
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("   {:7} | {:10} | {:15.2} | {:20.2}",
                num_threads, iterations, duration.as_secs_f64() * 1000.0, avg_ns);
        }
    }
}

/// 4: Atomic Operations
/// overhead of atomic fetch_add operations
fn atomic_benchmark() {
    println!("4. Atomic Operations Benchmark");
    println!("   Measures atomic fetch_add overhead");
    println!("   ------------------------------------------------");
    println!("   Threads | Iterations | Total Time (ms) | Avg Cost per Op (ns)");
    println!("   --------|------------|-----------------|---------------------");
    
    for &num_threads in THREAD_COUNTS {
        for &iterations in ITERATIONS {
            let counter = Arc::new(AtomicU64::new(0));
            let start = Instant::now();
            
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let counter_clone = Arc::clone(&counter);
                    thread::spawn(move || {
                        for _ in 0..iterations {
                            counter_clone.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let duration = start.elapsed();
            let total_ops = iterations * num_threads;
            let avg_ns = duration.as_nanos() as f64 / total_ops as f64;
            
            println!("   {:7} | {:10} | {:15.2} | {:20.2}",
                num_threads, iterations, duration.as_secs_f64() * 1000.0, avg_ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
