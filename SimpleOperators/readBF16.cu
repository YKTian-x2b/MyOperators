#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include "../Common.cuh"
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define GRD_SIZE 1
#define BLK_SIZE 32

__global__ void readBF16(float *out){
    int tid = threadIdx.x;
    uint16_t reg = 1.0;
    out[tid] = static_cast<float>(reg);
}

int main(){
    float *d_out;
    size_t Bytes = sizeof(float) * BLK_SIZE;
    CHECK(cudaMalloc(&d_out, Bytes));

    readBF16<<<GRD_SIZE, BLK_SIZE>>>(d_out);

    CHECK(cudaFree(d_out)); 
}

// nvcc --keep --keep-dir midRes -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -L /usr/local/cuda/lib64 -l cuda -o res/readBF16 readBF16.cu 
// cuasm --bin2asm midRes/readBF16.sm_86.cubin -o midRes/readBF16.sm_86.cuasm