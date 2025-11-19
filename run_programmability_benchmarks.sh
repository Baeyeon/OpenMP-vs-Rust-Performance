#!/bin/bash

# Programmability Benchmark (Prefix Sum)

set -e

echo "=== Programmability Benchmarks (Prefix Sum) ==="
echo ""
echo "Compiling OpenMP..."
cd openMP/src/programmability
gcc -O3 -march=native -fopenmp -std=c11 -o mp_prefix_sum prefix_sum.c
cd ../../..

echo "Running OpenMP..."
./openMP/src/programmability/mp_prefix_sum
echo ""
echo "Running Rust..."
cd rust
cargo build --release --bin prefix_sum 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin prefix_sum 2>/dev/null
cd ..
echo ""
