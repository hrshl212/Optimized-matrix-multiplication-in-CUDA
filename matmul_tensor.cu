#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda; // For WMMA

// Define WMMA tile size
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// CUDA Kernel for Tensor Core matrix multiplication
__global__ void matrixMulTensorCoreKernel(half* A, half* B, float* C, int N) {
    // Compute warp indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32; // Warp row
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // Warp col

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator with zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Compute global row/col indices
    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;

    // Loop over tiles in the K dimension
    for (int k = 0; k < N; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * N + k, N);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the result
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// Function to execute matrix multiplication on a single GPU
void gpuMatrixMulTensorCore(half* h_A, half* h_B, float* h_C, int N) {
    half *d_A, *d_B;
    float *d_C;
    size_t size = N * N * sizeof(half);
    size_t sizeC = N * N * sizeof(float);

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC);

    // Define CUDA kernel execution parameters
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / 16, N / 16);

    // Launch the kernel
    matrixMulTensorCoreKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 8192; // Matrix size (N x N)
    size_t size = N * N * sizeof(half);
    size_t sizeC = N * N * sizeof(float);

    // Allocate host memory
    half* h_A = (half*)malloc(size);
    half* h_B = (half*)malloc(size);
    float* h_C = (float*)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0;//static_cast<half>(rand()) / RAND_MAX;
        h_B[i] = 1.0;//static_cast<half>(rand()) / RAND_MAX;
    }

    // Perform matrix multiplication using Tensor Cores
    gpuMatrixMulTensorCore(h_A, h_B, h_C, N);

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Matrix multiplication completed successfully with Tensor Cores on a SINGLE GPU!" << std::endl;
    return 0;
}
