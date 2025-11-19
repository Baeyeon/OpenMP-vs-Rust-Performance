#!/bin/bash

# Benchmark Suite: Runtime Overhead, Programmability, and Scalability

set -e  

#==============================================================================
# Runtime Overhead Benchmark
#==============================================================================

echo "=== 1. Runtime Overhead Benchmarks ==="
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

#==============================================================================
# Programmability Benchmark
#==============================================================================

echo "=== 2. Programmability Benchmarks (Prefix Sum) ==="
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

#==============================================================================
# Scalability Benchmark
#==============================================================================

echo "=== 3. Scalability Benchmarks (Matrix Multiply) ==="
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

echo "=== All benchmarks completed! ==="
echo ""
echo "Output files:"
echo "  - runtime_overhead_results.csv"
echo "  - openmp_scalability_results.txt"
echo "  - rust_scalability_results.txt"
echo ""
