# OpenMP vs Rust Performance Comparison

A comprehensive comparison of programmability, scalability, runtime overhead, and control between OpenMP and Rust parallel computing approaches.

## Repository Structure

```
OpenMP-vs-Rust-Performance/
├── openMP/
│   └── src/
│       ├── control/              # Control comparison implementations
│       ├── programmability/      # Programmability benchmarks
│       ├── runtime_overhead/     # Runtime overhead tests
│       └── scalability/          # Scalability benchmarks
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── controllability/      # Control comparison implementations
│       ├── programmability/      # Programmability benchmarks
│       ├── runtime_overhead/     # Runtime overhead tests
│       └── scalability/          # Scalability benchmarks
├── run_control_benchmarks.sh
├── run_overhead_benchmarks.sh
├── run_programmability_benchmarks.sh
├── run_scalability_benchmarks.sh
└── README.md                          # Documentation files
```

## Running Benchmarks

Each benchmark script automatically compiles and runs both OpenMP and Rust implementations, before running any of the following script, make sure you are on crunchy1:

### 1. Control/Controllability Benchmarks
```bash
./run_control_benchmarks.sh
```
**Compilation:**
- OpenMP: `gcc -O3 -march=native -fopenmp -std=c11 control.c -o control_openmp`
- Rust: `cargo build --release --bin histogram`

**Output:** `controllability_results.csv`

Tests four control aspects: shared/private variables (atomic vs local), granularity (scheduling/chunk sizes), false sharing (padding), and thread affinity (core pinning).

### 2. Runtime Overhead Benchmarks
```bash
./run_overhead_benchmarks.sh
```
**Compilation:**
- OpenMP: `gcc -O3 -march=native -fopenmp -std=c11 overhead.c -o overhead_openmp`
- Rust: `cargo build --release --bin runtime_overhead`

**Output:** `runtime_overhead_results.csv`

Measures parallel runtime overhead with varying thread counts and iteration counts.

### 3. Programmability Benchmarks
```bash
./run_programmability_benchmarks.sh
```
**Compilation:**
- OpenMP: `gcc -O3 -march=native -fopenmp -std=c11 -o mp_prefix_sum prefix_sum.c`
- Rust: `cargo build --release --bin prefix_sum`

**Output:** Results printed to stdout

Implements parallel prefix sum to compare code complexity and ease of implementation.

### 4. Scalability Benchmarks
```bash
./run_scalability_benchmarks.sh
```
**Compilation:**
- OpenMP: `gcc -O3 -march=native -fopenmp -std=c11 -o mp_matrix_multiply matrix_multiply.c -lm`
- Rust: `cargo build --release --bin matrix_multiply`

**Output:** `openmp_scalability_results.txt` and `rust_scalability_results.txt`

Tests parallel matrix multiplication performance across different thread counts.
