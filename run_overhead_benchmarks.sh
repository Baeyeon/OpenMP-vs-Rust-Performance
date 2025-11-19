#!/bin/bash

# Runtime Overhead Benchmark

set -e  

echo "=== Runtime Overhead Benchmarks ==="
echo ""

OUTPUT_FILE="runtime_overhead_results.csv"
THREAD_COUNTS=(1 2 4 8 16)
ITERATIONS=(10000 25000 50000 75000 100000)

rm -f "$OUTPUT_FILE"

cd openMP/src/runtime_overhead
gcc -O3 -march=native -fopenmp -std=c11 overhead.c -o overhead_openmp
cd ../../..

echo "Running OpenMP benchmarks..."
for T in "${THREAD_COUNTS[@]}"; do
    for R in "${ITERATIONS[@]}"; do
        echo "  - T=$T, R=$R"
        ./openMP/src/runtime_overhead/overhead_openmp $T $R >> "$OUTPUT_FILE"
    done
done

echo ""
echo "Running Rust benchmarks..."
cd rust
cargo build --release --bin runtime_overhead 2>&1 | grep -v "Compiling\|Finished" || true
cargo run --release --bin runtime_overhead 2>/dev/null >> "../$OUTPUT_FILE"
cd ..

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
