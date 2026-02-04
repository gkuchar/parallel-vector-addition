/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

// GLOBAL VARIABLES
// Error code to check return values for CUDA calls
cudaError_t err = cudaSuccess;

// Timing metrics
cudaEvent_t start, stop;

// Print the vector length to be used, and compute its size
int    numElements = 100000000;
size_t size        = numElements * sizeof(float);

// Total running times across n executions to compute average.
float cpuReadbackTotal = 0;
float kernExecTotal = 0;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

void vecAddProcess() {
    // Allocate inputs and output on unified memory
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Verify that allocations succeeded
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Failed to allocate vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the input vectors
    for (int i = 0; i < numElements; ++i) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    // Prefetch memory onto GPU
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(A, size, device);
    cudaMemPrefetchAsync(B, size, device);
    cudaMemPrefetchAsync(C, size, device);
    cudaDeviceSynchronize();

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Record start time for kernal execution
    cudaEventRecord(start);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Record stop time for kernal execution
    cudaEventRecord(stop);
    // Synchronize
    cudaEventSynchronize(stop);

    // Save running time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Add running time to kernal execution total time
    kernExecTotal += milliseconds;

    // Synchronize unified memory and kernal
    cudaDeviceSynchronize();

    // Record start time for CPU read-beack
    cudaEventRecord(start);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Record stop time for CPU read-beack
    cudaEventRecord(stop);
    // Synchronize
    cudaEventSynchronize(stop);

    // Save running time
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Add running time to CPU read-back total time
    cpuReadbackTotal += milliseconds;

    printf("Test PASSED\n");

    // Free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

/**
 * Host main routine (modified for n iterations)
 */
int main(void) {

    // Generate number of iterations
    srand(time(NULL));
    int n = (rand() % 16) + 10;

    printf("\n[[[Vector addition of %d elements, %d iterations]]]\n\n", numElements, n);

    // Create events once
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    vecAddProcess();

    // Reset totals after warm-up
    cpuReadbackTotal = 0;
    kernExecTotal = 0;

    int i;
    for (i = 0; i < n; i++) {
        printf("START: iteration (%d/%d)\n", i + 1, n);
        vecAddProcess();
        printf("END: iteration (%d/%d)\n", i + 1, n);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nThe average CPU read-back / verification access time: %f milliseconds\n", cpuReadbackTotal / n);
    printf("The average kernel execution time: %f milliseconds\n", kernExecTotal / n);
    printf("The number of runs used to compute the averages: %d\n", n);

    return 0;
}