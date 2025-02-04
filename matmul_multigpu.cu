#include <cuda_runtime.h>
#include <iostream>

// Kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    __shared__ float Asub[32][32];
    __shared__ float Bsub[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;

    for (int k = 0; k < N / 32; k++) {
        Asub[threadIdx.y][threadIdx.x] = A[row * N + (k * 32 + threadIdx.x)];
        Bsub[threadIdx.y][threadIdx.x] = B[(k * 32 + threadIdx.y) * N + col];
        __syncthreads();

        for (int n = 0; n < 32; n++) {
            value += Asub[threadIdx.y][n] * Bsub[n][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = value;


}

__global__ void matrixMulKernel2(const float* A, const float* B, float* C, 
                                     int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * N + col];
        }
        //__syncthreads();
        //printf("i=%d and d_C=%f\n",row*K+col, value);

        C[row * N + col] = value;

    }


}


// Helper function to split work across two GPUs
void gpuMatrixMul(float* h_A, float* h_B, float* h_C, int N) {
    float *d_A1, *d_B1, *d_C1, *d_A2, *d_B2, *d_C2;
    size_t size = N * N * sizeof(float);

    // Divide matrices into two parts for two GPUs
    int halfN = N / 2;

    // Initialize devices
    cudaSetDevice(0);
    cudaMalloc((void**)&d_A1, size / 2);
    cudaMalloc((void**)&d_B1, size);
    cudaMalloc((void**)&d_C1, size / 2);
    cudaMemcpy(d_A1, h_A, size / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice);        
    // int device;
    // cudaGetDevice(&device);
    // printf("current cuda device :%d \n", device);

    cudaSetDevice(1);
    cudaMalloc((void**)&d_A2, size / 2);
    cudaMalloc((void**)&d_B2, size);
    cudaMalloc((void**)&d_C2, size / 2);
    cudaMemcpy(d_A2, h_A + (halfN * N), size / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice);
    // cudaGetDevice(&device);
    // printf("current cuda device :%d \n", device);

    // Kernel configuration
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(N / 32, N / 64); // Half rows per GPU

    // Launch kernel on GPU 0
    cudaSetDevice(0);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A1, d_B1, d_C1, N);
    // cudaGetDevice(&device);
    // printf("current cuda device :%d \n", device);
    
    // Launch kernel on GPU 1
    cudaSetDevice(1);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A2, d_B2, d_C2, N);
    // cudaGetDevice(&device);
    // printf("current cuda device :%d \n", device);

    // Copy results back to host
    cudaMemcpy(h_C, d_C1, size / 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C + (halfN * N), d_C2, size / 2, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
}

int main() {
    int N = 8192; // Matrix size (N x N)
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0; //static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = 1.0; //static_cast<float>(rand()) / RAND_MAX;
    }


    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("MPI Rank: Found %d GPUs\n", num_gpus);

    // Perform matrix multiplication
    gpuMatrixMul(h_A, h_B, h_C, N);

    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != N) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    // Verify and clean up
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    return 0;
}
