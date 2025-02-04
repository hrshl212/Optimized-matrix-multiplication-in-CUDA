# Optimized-matrix-matrix-multiplication-in-CUDA
The repository contains different ways to implement matrix-matrix multiplication in CUDA starting from basic implementation to using tensor cores in NVIDIA A100 GPUs. Two matrix sizes are considered first with 1024 x 1024 and second with 8192 x 8192. I check the timing for kernel execution or memory transfer using NSIGHT SYSTEM. Consedering max threds per streaming multiprocessor is usually 1024, I use a block size of 32 x 32 in all the codes.

## Basic Implementation (basic_matmul.cu)
The basic implementation of matrix-matrix multiplication (A x B = C) involves one thread multiplying one row of A with one column of B to get one value in C. It takes a kernel execution time of 9.6 ms whereas the memory transfer takes a time of 2.9 ms for the matrices of size 1024 x 1024. For 8192 x 8192 size, the kernel execution takes a time of 3213 ms and memory transfer takes a time of 214.4 ms. 

## Asynchronous Implementation (matmul_asynchronous.cu)
In this implementation CUDA stream is utilized to implement kernel execution and memory transfer concurrently. Stream 0 is utilized for memory transfer and Stream 1 is utlilized for kernel execution on same GPU. For 1024 x 1024, the kernel execution takes a time of 8.9 ms and memory transfer takes a time of 1.8 ms. Therefore, there is reduction in kernel execution time by `7.3%` and memory transfer time by `37.9%` with respect to the basic implementation. For 8192 x 8192 size, the kernel execution takes a time of 3265.6 ms and memory transfer takes a time of 203.9 ms. Changing tile size did lot lead to much improvement.

## Multi-GPU Implementation (matmul_multigpu.cu)

