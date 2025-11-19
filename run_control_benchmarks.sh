#!/bin/bash

# Controllability Benchmarks (Histogram)
# Tests: Shared/Private Variables, Granularity Control, False Sharing Control

set -e  

OUTPUT_FILE="controllability_results.csv"
THREAD_COUNTS=(1 2 4 8 16)
STRATEGIES=("atomic" "local")
DISTRIBUTIONS=("uniform" "skewed")
N=10000000  # 10 million elements

# OpenMP scheduling strategies
SCHEDULES=("static" "dynamic" "guided")
CHUNKS=(0 1000 10000)  # 0 = default

# Rayon grain sizes
GRAINS=(0 1000 10000)  # 0 = auto

# Padding options for false sharing control
PADS=(0 1)  # 0 = no padding, 1 = padded

echo "=============================================="
echo "Controllability Benchmarks"
echo "=============================================="
echo "Testing four aspects of control:"
echo "  1. Shared vs Private Variables (atomic vs local)"
echo "  2. Granularity Control (scheduling/chunk/grain)"
echo "  3. False Sharing Control (padding)"
echo "  4. Thread Affinity Control (thread-to-core pinning)"
echo ""
echo "N=$N elements"
echo ""
echo "OpenMP: Uses native proc_bind clause"
echo "Rust:   Uses core_affinity crate"
echo "=============================================="
echo ""

rm -f "$OUTPUT_FILE"

# Compile OpenMP version
echo "Compiling OpenMP version..."
cd openMP/src/control
gcc -O3 -march=native -fopenmp -std=c11 control.c -o control_openmp
cd ../../..

# ============================================
# OpenMP Benchmarks
# ============================================
echo ""
echo "=========================================="
echo "Running OpenMP Benchmarks"
echo "=========================================="

# 1. Shared vs Private: atomic vs local strategies
echo ""
echo "1. Testing Shared/Private Variable Control..."
for DIST in "${DISTRIBUTIONS[@]}"; do
    for T in "${THREAD_COUNTS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            echo "  OpenMP: $STRATEGY, dist=$DIST, T=$T (default settings)"
            ./openMP/src/control/control_openmp "$STRATEGY" "$DIST" "$N" "$T" "static" 0 0 >> "$OUTPUT_FILE"
        done
    done
done

# 2. Granularity Control: scheduling strategies and chunk sizes
echo ""
echo "2. Testing Granularity Control (OpenMP)..."
for DIST in "uniform"; do  # Use one dist to reduce combinations
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for SCHED in "${SCHEDULES[@]}"; do
            for CHUNK in "${CHUNKS[@]}"; do
                echo "  OpenMP atomic: sched=$SCHED, chunk=$CHUNK, T=$T"
                ./openMP/src/control/control_openmp "atomic" "$DIST" "$N" "$T" "$SCHED" "$CHUNK" 0 >> "$OUTPUT_FILE"
            done
        done
    done
done

# 3. False Sharing Control: padding
echo ""
echo "3. Testing False Sharing Control (OpenMP)..."
for DIST in "skewed"; do  # Skewed creates more contention
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for PAD in "${PADS[@]}"; do
            echo "  OpenMP atomic: pad=$PAD, T=$T, dist=$DIST"
            ./openMP/src/control/control_openmp "atomic" "$DIST" "$N" "$T" "static" 0 "$PAD" 0 >> "$OUTPUT_FILE"
        done
    done
done

# 4. Thread Affinity Control: thread-to-core pinning
echo ""
echo "4. Testing Thread Affinity Control (OpenMP)..."
for DIST in "uniform"; do
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for AFFINITY in "${PADS[@]}"; do  # 0 = no pinning, 1 = pinned
            echo "  OpenMP atomic: affinity=$AFFINITY, T=$T"
            ./openMP/src/control/control_openmp "atomic" "$DIST" "$N" "$T" "static" 0 0 "$AFFINITY" >> "$OUTPUT_FILE"
        done
    done
done

# ============================================
# Rust/Rayon Benchmarks
# ============================================
echo ""
echo "=========================================="
echo "Compiling Rust version..."
echo "=========================================="
cd rust
cargo build --release --bin histogram 2>&1 | grep -v "Compiling\|Finished" || true

echo ""
echo "=========================================="
echo "Running Rust/Rayon Benchmarks"
echo "=========================================="

# 1. Shared vs Private: atomic vs local strategies
echo ""
echo "1. Testing Shared/Private Variable Control..."
for DIST in "${DISTRIBUTIONS[@]}"; do
    for T in "${THREAD_COUNTS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            echo "  Rayon: $STRATEGY, dist=$DIST, T=$T (default settings)"
            cargo run --release --bin histogram -- "$STRATEGY" "$DIST" "$N" "$T" 0 0 2>/dev/null >> "../$OUTPUT_FILE"
        done
    done
done

# 2. Granularity Control: grain sizes
echo ""
echo "2. Testing Granularity Control (Rayon)..."
for DIST in "uniform"; do
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for GRAIN in "${GRAINS[@]}"; do
            echo "  Rayon atomic: grain=$GRAIN, T=$T"
            cargo run --release --bin histogram -- "atomic" "$DIST" "$N" "$T" "$GRAIN" 0 2>/dev/null >> "../$OUTPUT_FILE"
        done
    done
done

# 3. False Sharing Control: padding
echo ""
echo "3. Testing False Sharing Control (Rayon)..."
for DIST in "skewed"; do
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for PAD in "${PADS[@]}"; do
            echo "  Rayon atomic: pad=$PAD, T=$T, dist=$DIST"
            cargo run --release --bin histogram -- "atomic" "$DIST" "$N" "$T" 0 "$PAD" 0 2>/dev/null >> "../$OUTPUT_FILE"
        done
    done
done

# 4. Thread Affinity Control: thread-to-core pinning
echo ""
echo "4. Testing Thread Affinity Control (Rayon)..."
for DIST in "uniform"; do
    for T in "${THREAD_COUNTS[@]}"; do  # Use all thread counts for consistency
        for AFFINITY in "${PADS[@]}"; do  # 0 = no pinning, 1 = pinned
            echo "  Rayon atomic: affinity=$AFFINITY, T=$T"
            cargo run --release --bin histogram -- "atomic" "$DIST" "$N" "$T" 0 0 "$AFFINITY" 2>/dev/null >> "../$OUTPUT_FILE"
        done
    done
done

cd ..

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Benchmarks Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Total measurements: $(wc -l < "$OUTPUT_FILE")"
echo ""
echo "Sample results:"
head -30 "$OUTPUT_FILE"
echo "..."
echo ""
echo "Analysis areas:"
echo "  1. Shared vs Private: Compare 'atomic' vs 'local' strategies"
echo "  2. Granularity: Compare OpenMP sched/chunk vs Rayon grain"
echo "  3. False Sharing: Compare pad=0 vs pad=1 performance"
echo "  4. Thread Affinity: Compare affinity=0 vs affinity=1 performance"
echo ""
echo "Note: OpenMP uses native proc_bind(close) clause"
echo "      Rust uses external core_affinity crate"
echo ""
