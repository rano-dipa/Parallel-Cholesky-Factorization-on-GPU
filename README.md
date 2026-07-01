# Parallel Cholesky Factorization on GPU using CUDA

A CUDA implementation of **parallel Cholesky factorization** for Symmetric Positive Definite (SPD) matrices. This project was developed as the major project for a **Parallel Computing** course and demonstrates how GPU parallelism can significantly accelerate dense linear algebra computations.

## Overview

Cholesky factorization decomposes a symmetric positive definite matrix **A** into an upper triangular matrix **R** such that:

\[
A = R^T R
\]

While the traditional algorithm is inherently sequential, this project parallelizes the computationally intensive stages using NVIDIA CUDA to exploit GPU hardware.

The implementation separates the algorithm into multiple CUDA kernels, enabling efficient execution while preserving numerical correctness.

## Features

- CUDA implementation of Cholesky factorization
- Parallel row normalization
- Parallel trailing matrix update using 2D CUDA thread blocks
- Configurable 1D and 2D thread block sizes
- Dynamic grid computation
- Device-side pivot storage
- Verification against serial implementation
- Performance benchmarking across different matrix sizes and CUDA configurations

## Algorithm

The implementation divides Cholesky factorization into three CUDA kernels:

### 1. Diagonal Factorization

- Computes the square root of the pivot element
- Executed by a single CUDA thread
- Stores pivot values on the device for later use

### 2. Row Factorization

- Normalizes the current row using the pivot
- Uses 1D thread parallelism
- Each thread processes one matrix element

### 3. Trailing Matrix Update

- Updates the remaining upper triangular submatrix
- Uses 2D CUDA thread blocks
- Dominates overall runtime
- Exploits fine-grained GPU parallelism

## Optimizations

Several improvements were implemented over the baseline version:

- Device-side pivot array
- Correct device row pointer initialization
- Safe parallel row normalization
- Upper-triangular-only updates
- Dynamic grid sizing
- Boundary checking
- Shared memory usage during matrix updates
- Host synchronization between kernel launches

## Experimental Evaluation

Experiments were performed on the **GRACE cluster** to evaluate correctness and performance.

### Experiment 1 — Matrix Size Scaling

Matrix sizes tested:

```
32
64
128
256
512
1024
2048
4096
```

Results showed:

- Zero numerical error
- Increasing GPU utilization for larger matrices
- Maximum speedup exceeding **11,000×** for 4096 × 4096 matrices

---

### Experiment 2 — Effect of 1D Block Size

Tested block sizes:

```
32
64
128
256
512
1024
```

Observations:

- Small block sizes underutilize the GPU
- Best performance achieved around **256–512 threads**
- Larger values provide diminishing returns

---

### Experiment 3 — Effect of 2D Block Size

Configurations evaluated:

```
1×1024
2×512
4×256
8×128
16×64
32×32
64×16
128×8
256×4
512×2
1024×1
```

Observations:

- Balanced thread blocks significantly improve performance
- Best results obtained around:
  - 32×32
  - 128×8
- Highly skewed block shapes reduce occupancy and memory efficiency

## Performance Highlights

- Correctness verified against the serial implementation
- Zero numerical error across all experiments
- Speedup scales with matrix size
- More than **11,000× acceleration** achieved on large matrices
- Balanced 2D CUDA blocks provide optimal performance

## Build

Load CUDA and compiler modules:

```bash
module load intel/2023a
module load CUDA/12.2
```

Compile:

```bash
nvcc -o parallel_cholesky parallel_cholesky.cu
```

## Usage

```bash
./parallel_cholesky <matrix_size> <1D_block_size> <block_x> <block_y>
```

Example:

```bash
./parallel_cholesky 1024 256 32 32
```

## Project Structure

```
.
├── parallel_cholesky.cu    # CUDA implementation
├── Report.pdf              # Project report
└── README.md
```

## Learning Outcomes

This project demonstrates:

- CUDA kernel design
- GPU thread hierarchy
- Parallel dense linear algebra
- Shared memory optimization
- Memory coalescing
- CUDA synchronization
- GPU performance tuning
- Experimental performance analysis

## Technologies

- CUDA
- NVIDIA GPU Computing
- C++
- NVCC
- Parallel Computing
- Dense Linear Algebra

## Results

The project demonstrates that GPU acceleration can dramatically reduce the execution time of Cholesky factorization while maintaining numerical correctness. Performance improvements become increasingly significant as matrix size grows, highlighting the effectiveness of CUDA-based parallelization for computationally intensive linear algebra workloads.

## Author

**Dipanwita Rano**

Parallel Computing Course Project
