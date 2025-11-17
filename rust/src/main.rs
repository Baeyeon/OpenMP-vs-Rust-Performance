// Main benchmark runner for all tests
// Allows running individual benchmarks or all benchmarks

use std::env;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    match args[1].as_str() {
        "programmability" => run_programmability_benchmarks(),
        "scalability" => run_scalability_benchmarks(),
        "runtime_overhead" => run_runtime_overhead_benchmarks(),
        "controllability" => run_controllability_benchmarks(),
        "all" => {
            run_programmability_benchmarks();
            println!("\n\n");
            run_scalability_benchmarks();
            println!("\n\n");
            run_runtime_overhead_benchmarks();
            println!("\n\n");
            run_controllability_benchmarks();
        },
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            println!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("Rust Benchmark Suite for OpenMP vs Rust Comparison");
    println!();
    println!("Usage: cargo run --release --bin run_all_benchmarks <command>");
    println!();
    println!("Commands:");
    println!("  programmability  - Run prefix sum benchmark (measures code complexity)");
    println!("  scalability      - Run matrix multiply benchmark (measures scalability)");
    println!("  runtime_overhead - Run runtime overhead benchmarks (thread operations & sync)");
    println!("  controllability  - Run histogram benchmark (measures programmer control)");
    println!("  all              - Run all benchmarks");
    println!("  help             - Show this help message");
    println!();
    println!("You can also run individual benchmarks directly:");
    println!("  cargo run --release --bin prefix_sum");
    println!("  cargo run --release --bin matrix_multiply [n] [threads]");
    println!("  cargo run --release --bin runtime_overhead");
    println!("  cargo run --release --bin histogram");
}

fn run_programmability_benchmarks() {
    println!("Running Programmability Benchmarks...");
    println!("=====================================\n");
    
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", "prefix_sum"])
        .status()
        .expect("Failed to run prefix_sum benchmark");
    
    if !status.success() {
        eprintln!("Prefix sum benchmark failed!");
    }
}

fn run_scalability_benchmarks() {
    println!("Running Scalability Benchmarks...");
    println!("==================================\n");
    
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", "matrix_multiply"])
        .status()
        .expect("Failed to run matrix_multiply benchmark");
    
    if !status.success() {
        eprintln!("Matrix multiply benchmark failed!");
    }
}

fn run_runtime_overhead_benchmarks() {
    println!("Running Runtime Overhead Benchmarks...");
    println!("======================================\n");
    
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", "runtime_overhead"])
        .status()
        .expect("Failed to run runtime_overhead benchmark");
    
    if !status.success() {
        eprintln!("Runtime overhead benchmark failed!");
    }
}

fn run_controllability_benchmarks() {
    println!("Running Controllability Benchmarks...");
    println!("=====================================\n");
    
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", "histogram"])
        .status()
        .expect("Failed to run histogram benchmark");
    
    if !status.success() {
        eprintln!("Histogram benchmark failed!");
    }
}
