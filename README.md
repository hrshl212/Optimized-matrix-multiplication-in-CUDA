# Optimized-matrix-matrix-multiplication-in-CUDA
The repository contains different ways to implement matrix-matrix multiplication in CUDA starting from basic implementation to using tensor cores in NVIDIA A100 GPUs. Two matrix sizes are considered first with 1024 x 1024 and second with 8192 x 8192. I check the timing for kernel execution or memory transfer using NSIGHT SYSTEM. Consedering max threds per streaming multiprocessor is usually 1024, I use a block size of 32 x 32 in all the codes.

# Basic Implementation (basic_matmul.cu)
The basic implementation of matrix-matrix multiplication (A x B = C) involves one thread multiplying one row of A with one column of B to get one value in C. It takes a kernel execution time of 9.6 ms whereas the memory transfer takes a time of 2.9 ms for the matrices of size 1024 x 1024. For 8192 x 8192 size, the kernel execution takes a time of 3213 ms and memory transfer takes a time of 214.4 ms. 


