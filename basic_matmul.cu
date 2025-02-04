#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, 
                                     int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float value = 0;
        for (int i = 0; i < M; ++i) {
            value += A[row * M + i] * B[i * K + col];
        }
        //__syncthreads();
        //printf("i=%d and d_C=%f\n",row*K+col, value);

        C[row * K + col] = value;

    }


}

void MatrixMultiply(const float* h_A, const float* h_B, float* h_C, 
                         int N, int M, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // cudaMalloc(&d_A, sizeA);
    // cudaMalloc(&d_B, sizeB);
    // cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A , h_A , 
                    N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Asynchronous memory copy for the entire B matrix

    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
                    (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel for the current tile
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A , 
                                                d_B, 
                                                d_C, 
                                                N, M, K);

    // Asynchronously copy the result back to host
    cudaMemcpy(h_C, d_C , 
                    N * K * sizeof(float), cudaMemcpyDeviceToHost);


    // Synchronize streams to ensure all operations complete
    cudaDeviceSynchronize();


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 8192, M = 8192, K = 8192;
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A, *h_B, *h_C;


	h_A = (float *)malloc(sizeA);
	h_B = (float *)malloc(sizeB);
	h_C = (float *)malloc(sizeC);

    // float *h_A = new float[N * M];
    // float *h_B = new float[M * K];
    // float *h_C = new float[N * K];

    // Initialize input matrices
    for (int i = 0; i < N * M; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < M * K; ++i) h_B[i] = 1.0f;

    MatrixMultiply(h_A, h_B, h_C, N, M, K);

    // for (int i = 0; i < M * K; ++i) {
    //     std::cout << "i=" << i <<", h_B=" << h_B[i] << std::endl;
    // }


    // for (int i = 0; i < N * K; ++i) {
    //     std::cout << "i=" << i <<", h_C=" << h_C[i] << std::endl;
    // }

    // Validate results
    for (int i = 0; i < N * K; ++i) {
        if (h_C[i] != M) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    std::cout << "code executed with success" << std::endl;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
