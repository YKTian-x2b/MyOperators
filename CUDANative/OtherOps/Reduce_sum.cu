#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include "helper_cuda.h"
#include "hostptr.hpp"
#include "cuptr.hpp"

#define BLK_SIZE 256
#define UNROLL 4

#define N_DATA 1024

float A[N_DATA];

void assignData(){
    for(int j = 0; j < N_DATA; j++){
        A[j] = j;
    }
}

__forceinline__ __device__ float warpShuffle(float localSum){
    localSum += __shfl_xor_sync(uint(-1), localSum, 16);
    localSum += __shfl_xor_sync(uint(-1), localSum, 8);
    localSum += __shfl_xor_sync(uint(-1), localSum, 4);
    localSum += __shfl_xor_sync(uint(-1), localSum, 2);
    localSum += __shfl_xor_sync(uint(-1), localSum, 1);
    return localSum;
}

__global__ void reduce_sum(float *in, float *out){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdx = blockDim.x;
    int idx = bid * bdx * UNROLL + tid;
    __shared__ float data_out[BLK_SIZE];

    float sum = 0;
    #pragma unroll
    for(int i = idx, j = 0; i < N_DATA, j<UNROLL; i+=bdx,j++){
        sum += in[i];
    }
    data_out[tid] = sum;
    __syncthreads();

    if(bdx >= 1024 && tid < 512){
        data_out[tid] += data_out[tid+ 512];
    }
    __syncthreads();
    if(bdx >= 512 && tid < 256){
        data_out[tid] += data_out[tid+ 256];
    }
    __syncthreads();
    if(bdx >= 256 && tid < 128){
        data_out[tid] += data_out[tid+ 128];
    }
    __syncthreads();
    if(bdx >= 128 && tid < 64){
        data_out[tid] += data_out[tid+ 64];
    }
    __syncthreads();
    if(bdx >= 64 && tid < 32){
        data_out[tid] += data_out[tid+ 32];
    }
    __syncthreads();

    if (tid < 32) {
        data_out[tid] = warpShuffle(data_out[tid]);
    }
    if(tid == 0){
        out[tid] = data_out[tid];
    }
}

int main(){
    assignData();
    CuPtr<float> in_d(N_DATA);
    CuPtr<float> out_d(1);
    checkCudaErrors(cudaMemcpy(in_d.GetPtr(), A, N_DATA*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grd_size((N_DATA-1)/(BLK_SIZE*UNROLL) + 1);
    reduce_sum<<<grd_size, BLK_SIZE>>>(in_d.GetPtr(), out_d.GetPtr());
    cudaDeviceSynchronize();

    HostPtr<float> out_h;
    out_d.ToHostPtr(out_h);
    std::cout << out_h(0) << std::endl;
}

// nvcc -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/reduce_sum Reduce_sum.cu