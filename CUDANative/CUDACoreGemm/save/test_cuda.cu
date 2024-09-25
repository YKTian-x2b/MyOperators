#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <ctime>
#include "../Common.cuh"

#define BDIMX 32

__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);
    return localSum;
}

__global__ void testShuffle(int * input, int * output){
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = tid;
    int mySum = input[idx];
    mySum = warpReduce(mySum);
    output[idx] = mySum;
}

int main() {
    int size_bytes = sizeof(int) * BDIMX;
    int * h_in = (int*)malloc(size_bytes);
    int * h_out = (int*)malloc(size_bytes);
    for(int i = 0; i < BDIMX; i++){
        h_in[i] = i + 1;
    }
    int * d_in, *d_out;
    cudaMalloc(&d_in, size_bytes);
    cudaMalloc(&d_out, size_bytes);

    cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice);

    testShuffle<<<1, BDIMX>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, size_bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < BDIMX; i++){
        printf("%d ", h_out[i]);
    }
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}