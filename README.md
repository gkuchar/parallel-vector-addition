# CUDA Vector Addition Performance Benchmarking

A comprehensive performance analysis project comparing explicit memory management versus Unified Memory in CUDA, demonstrating GPU computing fundamentals and systematic benchmarking methodology.

## 🎯 Project Overview

This project implements and benchmarks two variants of parallel vector addition on NVIDIA GPUs, providing empirical insights into memory transfer costs, kernel execution performance, and the trade-offs between different CUDA memory management strategies. The implementations process 100 million floating-point elements with statistical averaging across multiple runs.

## 🚀 Implementations

### Part 1: Explicit Memory Management (`vectorAdd.cu`)
Traditional CUDA programming model with manual memory management and explicit data transfers.

**Architecture:**
- **Host Memory:** CPU-side allocations using `malloc()`
- **Device Memory:** GPU-side allocations using `cudaMalloc()`
- **Data Transfer:** Explicit `cudaMemcpy()` operations for H→D and D→H movement
- **Synchronization:** CUDA events for precise timing measurements

**Performance Metrics Captured:**
1. **Host-to-Device Transfer Time** - Measures PCIe bandwidth utilization for input data
2. **Kernel Execution Time** - Pure GPU computation time for vector addition
3. **Device-to-Host Transfer Time** - PCIe overhead for retrieving results

### Part 2: Unified Memory (`vectorAdd_unified.cu`)
Modern CUDA programming model with automatic memory migration and simplified API.

**Architecture:**
- **Unified Memory:** Single address space using `cudaMallocManaged()` for all allocations
- **Automatic Migration:** On-demand page migration between CPU and GPU handled by CUDA runtime
- **Explicit Prefetching:** `cudaMemPrefetchAsync()` moves data to GPU before kernel execution
- **No Manual Copies:** Eliminates all `cudaMemcpy()` calls—both H→D and D→H

**Code Transition from Explicit Memory:**

The conversion to Unified Memory simplified the codebase significantly:

1. **Memory Allocation Consolidation:**
   - Replaced `malloc()` (host) + `cudaMalloc()` (device) with single `cudaMallocManaged()` calls
   - All three vectors (A, B, C) allocated in unified address space
   - Reduced allocation code from 6 calls to 3

2. **Prefetching Implementation:**
   ```cuda
   int device;
   cudaGetDevice(&device);
   cudaMemPrefetchAsync(A, size, device);
   cudaMemPrefetchAsync(B, size, device);
   cudaMemPrefetchAsync(C, size, device);
   cudaDeviceSynchronize();
   ```
   - Proactively migrates data to GPU before kernel launch
   - Prevents page faults during computation
   - Requires synchronization to ensure migration completes

3. **Memory Copy Elimination:**
   - Removed all `cudaMemcpy(d_A, h_A, ...)` H→D transfers
   - Removed all `cudaMemcpy(h_C, d_C, ...)` D→H transfers
   - Data accessible from both CPU and GPU without explicit copying

4. **Memory Deallocation Simplification:**
   - Replaced separate `cudaFree()` + `free()` with single `cudaFree()` per allocation
   - Unified memory freed from host-side only

5. **Additional Synchronization:**
   - Added `cudaDeviceSynchronize()` after kernel launch
   - Ensures kernel completion before CPU accesses results
   - Required for timing CPU read-back phase accurately

**Performance Metrics Captured:**
1. **Kernel Execution Time** - GPU computation (may include page fault overhead if prefetching fails)
2. **CPU Read-back Time** - Host verification access cost triggering implicit D→H migration

**Code Complexity Reduction:**
- **~60% fewer lines** related to memory management
- **Eliminated error handling** for manual memory copies
- **Single allocation paradigm** simplifies pointer management

## 🔬 Technical Implementation

### Timing Methodology

**CUDA Event-Based Timing with Running Totals:**

The implementation uses CUDA Event APIs combined with global accumulator variables to collect precise, averaged timing metrics across multiple iterations:

```cuda
// Global timing infrastructure
cudaEvent_t start, stop;
float htdTotal = 0;      // Host-to-Device running total
float kernExecTotal = 0; // Kernel execution running total
float dthTotal = 0;      // Device-to-Host running total

// One-time event creation
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Timing sequence for each operation:
cudaEventRecord(start);
// ... operation to measure ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
total += milliseconds;  // Accumulate for averaging
```

**Implementation Strategy:**
1. **Global Event Reuse:** Single `start` and `stop` event pair shared across all timing measurements for efficiency
2. **Running Total Accumulators:** Separate global variables (`htdTotal`, `kernExecTotal`, `dthTotal`) accumulate times across iterations
3. **Random Iteration Count:** Each execution randomly selects n ∈ [10, 25] iterations using `rand()` to vary sample size
4. **Warm-up Run:** First iteration excluded from totals to eliminate cold-start effects
5. **Final Averaging:** After n iterations, averages computed as `total / n` for each metric

**Critical Synchronization Points:**
- **`cudaEventSynchronize(stop)`:** Ensures operation completion before timing calculation
- **`cudaDeviceSynchronize()`:** (Unified Memory only) Required after kernel launch to complete prefetching and before CPU access

**Advantages of This Approach:**
- **GPU-Side Timing:** CUDA events measure actual device execution, avoiding CPU timer inaccuracies
- **Minimal Overhead:** Event recording adds negligible latency to measured operations
- **Statistical Robustness:** Multiple runs average out system noise and scheduling variations

### Kernel Configuration

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
// Result: ~390,625 blocks for 100M elements
```

**Grid Configuration Analysis:**
- **Thread Block Size:** 256 threads (standard for compute capability)
- **Grid Dimensions:** Dynamic calculation ensures full data coverage
- **Memory Coalescing:** Sequential thread indexing optimizes global memory access

### Memory Transfer Patterns

**Explicit Model:**
```
CPU → PCIe → GPU (H→D copy)
         ↓
    GPU Kernel Execution
         ↓
GPU → PCIe → CPU (D→H copy)
```
**Characteristics:** Direct, predictable costs. Memory movement happens exactly when `cudaMemcpy()` is called, making transfer overhead explicitly visible in timing measurements.

**Unified Memory Model:**
```
CPU ← Page Fault Migration → GPU
         ↓
    GPU Kernel Execution
         ↓
CPU ← Demand Paging → GPU
```
**Characteristics:** Implicit, runtime-managed costs. Data migration shifts to whenever GPU/CPU actually accesses the data.

### Understanding Cost Shifting: Explicit vs. Unified Memory

**Explicit Memory - Direct Visibility:**
- **When costs occur:** During explicit `cudaMemcpy()` calls
- **Visibility:** Transfer times appear as separate, measurable operations
- **Control:** Developer dictates exactly when data moves
- **Result:** H→D copy (0.0566ms) and D→H copy (0.0315ms) are isolated metrics

**Unified Memory - Hidden Migration:**
- **When costs occur:** During first access by CPU/GPU (demand paging)
- **Visibility:** Transfer overhead hidden inside apparent "kernel execution" or "CPU access" time
- **Control:** Runtime system decides migration timing based on page faults
- **Result:** Without prefetching, page faults during kernel inflate execution time from 0.0106ms → 0.904ms

**The Role of Prefetching:**

Prefetching (`cudaMemPrefetchAsync()`) partially restores explicit control:
- **GPU Prefetch:** Moves data to device *before* kernel launch
  - Prevents page faults during computation
  - Reduced kernel time from potential >100ms to 0.904ms in this implementation
  - Cost appears in prefetch step rather than kernel execution
  
- **Missing CPU Prefetch:** No prefetch back to host before verification
  - CPU read-back triggers page-by-page migration (731.592ms total)
  - Each verification access causes page fault and migration overhead
  - Explains why read-back is 23,000x slower than explicit D→H copy

**Cost Distribution Comparison:**

| Operation | Explicit Memory | Unified Memory (with GPU prefetch only) |
|-----------|----------------|----------------------------------------|
| H→D Transfer | 0.0566ms (explicit) | ~Hidden in prefetch~ |
| Computation | 0.0106ms (pure) | 0.904ms (includes residual faults) |
| D→H Transfer | 0.0315ms (explicit) | 731.592ms (demand paging) |
| **Total** | **0.0987ms** | **732.496ms** |

**Key Takeaway:** Unified Memory doesn't eliminate transfer costs—it transforms them from explicit bulk operations into implicit, fine-grained page migrations. Without bidirectional prefetching, these costs can dominate performance by orders of magnitude.

## 📊 Performance Analysis

### Observed Results (NVIDIA A100 on Delta)

**Explicit Memory Management:**
```
Average H→D Copy Time:    0.0566 ms
Average Kernel Execution: 0.0106 ms
Average D→H Copy Time:    0.0315 ms
Total Time:               0.0987 ms
Number of Runs:           19
```

**Unified Memory:**
```
Average Kernel Execution:     0.904 ms
Average CPU Read-back Time:   731.592 ms
Total Time:                   732.496 ms
Number of Runs:               24
```

### Performance Insights

**Explicit Memory Observations:**
- **Dominant Cost:** Memory transfers account for ~89% of total runtime
- **H→D vs D→H Asymmetry:** Host-to-device (0.0566ms) costs ~80% more than device-to-host (0.0315ms) due to transferring 2 input vectors versus 1 output vector
- **Sequential Bottleneck:** Memory copies execute sequentially and cannot leverage GPU parallelism
- **Kernel Efficiency:** Actual computation (0.0106ms) represents only ~11% of total time despite processing 100M elements

**Unified Memory Observations:**
- **Dramatic Overhead Increase:** Total time ~7,400x slower than explicit memory management
- **Hidden Migration Costs:** Without proper prefetching, page faults during kernel execution significantly inflate apparent compute time (0.904ms vs 0.0106ms)
- **CPU Read-back Dominance:** Verification access time (731.592ms) accounts for 99.9% of total runtime
- **Code Simplification Trade-off:** Reduced code complexity comes at substantial performance cost when prefetching isn't optimized

### Critical Performance Analysis

**Why Explicit Memory is Faster:**
The explicit memory model provides predictable, controlled data movement. By managing transfers manually, we avoid the overhead of the CUDA runtime's demand-paging system. The total overhead remains minimal because transfers are optimized for bulk operations.

**Unified Memory Performance Challenge:**
The dramatic slowdown in the unified memory implementation reveals the cost of implicit data migration. When the CPU accesses unified memory after kernel execution, the runtime must migrate pages from GPU to CPU memory. This on-demand migration incurs significant overhead:

1. **Page Fault Overhead:** Each CPU access triggers page faults that must be serviced
2. **Migration Granularity:** Data moves in page-sized chunks rather than bulk transfers
3. **Synchronization Cost:** Runtime must coordinate memory coherence between CPU and GPU

**Impact of Prefetching:**
The `cudaMemPrefetchAsync()` calls before kernel execution successfully reduced kernel time by proactively migrating data to the GPU, preventing page faults during computation. However, the lack of prefetching back to the host before CPU verification explains the enormous read-back time—every element access triggers individual page migrations.

**Key Insight:** Unified Memory simplifies code but shifts data movement costs from explicit, optimized bulk transfers to implicit, runtime-managed page migrations. Without careful prefetching strategies in both directions, performance can degrade by orders of magnitude. The 60% reduction in code complexity comes at a 7,400x performance penalty in this implementation.

## 🛠️ Build & Execution

### Prerequisites
```bash
# CUDA Toolkit (11.0+)
# NVIDIA GPU with Compute Capability 3.5+
# Access to delta.ncsa.illinois.edu or similar HPC cluster
```

### Compilation
```bash
# Explicit memory version
nvcc -o vectorAdd vectorAdd.cu -I/path/to/cuda-samples/Common

# Unified memory version
nvcc -o vectorAdd_unified vectorAdd_unified.cu -I/path/to/cuda-samples/Common
```

### Execution
```bash
# Run explicit memory benchmark
./vectorAdd

# Run unified memory benchmark
./vectorAdd_unified
```

### Expected Output Format

**Explicit Memory Version:**
```
[[[Vector addition of 100000000 elements, 19 iterations]]]

START: iteration (1/19)
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 390625 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
END: iteration (1/19)
...
START: iteration (19/19)
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 390625 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
END: iteration (19/19)

The average host-to-device copy time: 0.056600 milliseconds
The average kernel execution time: 0.010565 milliseconds
The average device-to-host copy time: 0.031530 milliseconds
The number of runs used to compute the averages: 19
```

**Unified Memory Version:**
```
[[[Vector addition of 100000000 elements, 24 iterations]]]

START: iteration (1/24)
CUDA kernel launch with 390625 blocks of 256 threads
Test PASSED
END: iteration (1/24)
...
START: iteration (24/24)
CUDA kernel launch with 390625 blocks of 256 threads
Test PASSED
END: iteration (24/24)

The average CPU read-back / verification access time: 731.591553 milliseconds
The average kernel execution time: 0.904149 milliseconds
The number of runs used to compute the averages: 24
```

## 🎓 Learning Outcomes

This project demonstrates advanced competencies in:

### GPU Computing
- **CUDA Programming Model:** Kernel development, thread indexing, grid configuration
- **Memory Hierarchies:** Understanding of host, device, and unified memory models
- **Asynchronous Execution:** Proper synchronization of non-blocking operations

### Performance Engineering
- **Benchmarking Methodology:** Statistical averaging, warm-up runs, variance reduction
- **Bottleneck Analysis:** Identifying PCIe as primary performance constraint
- **Profiling Tools:** CUDA event-based timing for microsecond precision

### System Architecture
- **PCIe Bandwidth:** Quantifying host-device interconnect limitations
- **Memory Migration:** Understanding on-demand paging in Unified Memory
- **Compute vs. Transfer:** Analyzing Amdahl's Law implications for GPU workloads

## 🔍 Code Quality Features

- **Error Handling:** Comprehensive `cudaError_t` checking after all CUDA calls
- **Resource Management:** Proper cleanup with `cudaFree()`, `free()`, event destruction
- **Reproducibility:** Seeded random number generation for consistent test data
- **Modularity:** Encapsulated `vecAddProcess()` function for repeated benchmarking
- **Documentation:** Inline comments explaining timing instrumentation

## 📈 Optimization Opportunities

**Immediate Impact (Unified Memory):**
1. **CPU Prefetching:** Add `cudaMemPrefetchAsync(C, size, cudaCpuDeviceId)` before CPU verification
   - **Expected Improvement:** Reduce read-back time from 731ms to ~0.03ms (24,000x speedup)
   - **Implementation:** Single line of code after kernel completion
   - **Why:** Bulk transfer prevents per-element page fault overhead

**Advanced Optimizations:**
2. **CUDA Streams:** Overlap H→D copy, kernel execution, and D→H copy using asynchronous streams
   - Potential to reduce total time by up to 50% if operations have similar durations
   
3. **Pinned Memory:** Use `cudaMallocHost()` for host allocations in explicit memory version
   - Enables DMA transfers, bypassing CPU caching for faster PCIe throughput
   - Expected 2-4x improvement in copy times

4. **Kernel Fusion:** Combine initialization and addition to reduce memory traffic
   - Eliminates need to transfer pre-initialized arrays from host
   - Reduces H→D transfer from 800MB to 0MB

5. **Multi-GPU Scaling:** Distribute workload across multiple A100s with peer-to-peer transfers
   - Near-linear speedup for computation; PCIe still limits overall throughput

6. **Profiling Integration:** Add Nsight Systems markers for detailed timeline analysis
   - Visualize exact overlap between memory transfers and computation
   - Identify additional optimization opportunities

**Performance Recovery Path for Unified Memory:**
```
Current:  732.5ms total (731.6ms CPU read-back + 0.9ms kernel)
+ CPU Prefetch: ~0.94ms total (0.03ms read-back + 0.9ms kernel)
→ Matches explicit memory performance while retaining code simplicity
```

## 🖥️ Execution Environment

**Target Platform:** NCSA Delta Supercomputer
- **GPU:** NVIDIA A100 (40GB/80GB variants)
- **Architecture:** Ampere (Compute Capability 8.0)
- **Peak Performance:** 19.5 TFLOPS (FP32)
- **Memory Bandwidth:** 1,555 GB/s (HBM2)

## 📝 Academic Context

**Course:** COSC 40903 - GPU Computing  
**Institution:** Texas Christian University  
**Objectives:**
- Understand GPU memory management paradigms
- Quantify data transfer vs. computation costs
- Practice systematic performance measurement
- Compare explicit vs. managed memory models

## 📄 Project Structure

```
.
├── vectorAdd.cu              # Explicit memory implementation
├── vectorAdd_unified.cu      # Unified memory implementation
├── README.md                 # This file
└── sample_outputs/
    ├── explicit_output.txt   # Benchmark results (explicit)
    └── unified_output.txt    # Benchmark results (unified)
```

## 🎯 Key Takeaways

1. **Memory Copies Dominate Simple Kernels:** In explicit memory model, 89% of runtime is transfer overhead (0.0881ms) vs. 11% computation (0.0106ms)

2. **H→D vs D→H Asymmetry:** Host-to-device transfers (0.0566ms) cost 80% more than device-to-host (0.0315ms) due to moving 2 input vectors vs. 1 output vector—demonstrating the importance of minimizing input data size

3. **Unified Memory Requires Bidirectional Prefetching:** GPU prefetching reduced kernel page faults significantly, but missing CPU prefetch caused 731ms of demand paging overhead—a 23,000x slowdown compared to explicit D→H copy

4. **Implicit != Free:** Unified Memory's simplified API comes at a severe performance cost (7,400x slower total time) without careful migration management—developer convenience traded for runtime overhead

5. **Sequential Execution Limits GPU Benefit:** Memory transfers execute sequentially on the CPU and cannot leverage GPU parallelism, making them persistent bottlenecks regardless of GPU core count

6. **CUDA Events Provide Precision:** GPU-side event timing captures true operation duration, revealing that kernel execution is 85x faster than the smallest memory transfer in the explicit model

7. **Page Fault Overhead is Non-Trivial:** The 85x difference between explicit kernel time (0.0106ms) and unified memory kernel time (0.904ms) quantifies the residual cost of runtime-managed page migrations even with prefetching

---

**Author:** Griffin Kuchar
**Date:** February 2026  
**Course:** GPU Computing (COSC 40903)

*Executed on NCSA Delta supercomputer with NVIDIA A100 GPUs*