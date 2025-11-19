#!/bin/bash

# Scalability Benchmark (Matrix Multiply)

set -e

echo "=== Scalability Benchmarks (Matrix Multiply) ==="
echo ""
echo "Compiling OpenMP..."
cd openMP/src/scalability
gcc -O3 -march=native -fopenmp -std=c11 -o mp_matrix_multiply matrix_multiply.c -lm
cd ../../..

echo "Running OpenMP..."
./openMP/src/scalability/mp_matrix_multiply | tee openmp_scalability_results.txt
echo ""
echo "Running Rust..."
cd rust
cargo build --release --bin matrix_multiply 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin matrix_multiply 2>/dev/null | tee ../rust_scalability_results.txt
cd ..
echo ""

echo "=== Scalability benchmarks completed! ==="
echo ""
echo "Output files:"
echo "  - openmp_scalability_results.txt"
echo "  - rust_scalability_results.txt"
echo ""
