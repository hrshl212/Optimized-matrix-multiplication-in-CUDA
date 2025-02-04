#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 4096  // Define tile size for memory and computation

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

void tiledMatrixMultiply(const float* h_A, const float* h_B, float* h_C, 
                         int N, int M, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Divide matrices into tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;  //(1024+256-1)/256 == 4
    for (int tile = 0; tile < numTiles; ++tile) {
        int startRow = tile * TILE_SIZE;
        int numRows = min(TILE_SIZE, N - startRow);

        // Asynchronous memory copy for the current tile
        cudaMemcpyAsync(d_A + startRow * M, h_A + startRow * M, 
                        numRows * M * sizeof(float), cudaMemcpyHostToDevice, streams[0]);

        // Asynchronous memory copy for the entire B matrix
        if (tile == 0) {
            cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, streams[1]);
        }

        // Synchronize streams before computation starts
        cudaStreamSynchronize(streams[0]);
        //cudaStreamSynchronize(streams[1]);

        dim3 blockDim(32, 32);
        dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
                     (numRows + blockDim.y - 1) / blockDim.y);

        // Launch kernel for the current tile
        matrixMultiplyKernel<<<gridDim, blockDim, 0, streams[1]>>>(d_A + startRow * M, 
                                                                   d_B, 
                                                                   d_C + startRow * K, 
                                                                   numRows, M, K);

        //cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        // Asynchronously copy the result back to host
        cudaMemcpyAsync(h_C + startRow * K, d_C + startRow * K, 
                        numRows * K * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);

        // if(tile==0){
        //     for( int i=0; i< numRows*K; i++){
        //         printf("i=%d and hC=%.2f\n", i, h_C[i]);
        //     }
        // }
    }

    // Synchronize streams to ensure all operations complete
    cudaDeviceSynchronize();

    for (int i = 0; i < 2; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 8192, M = 8192, K = 8192;
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A = new float[N * M];
    float *h_B = new float[M * K];
    float *h_C = new float[N * K];

    // Initialize input matrices
    for (int i = 0; i < N * M; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < M * K; ++i) h_B[i] = 1.0f;

    tiledMatrixMultiply(h_A, h_B, h_C, N, M, K);

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
