#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include "helper_cuda.h"
#include "hostptr.hpp"
#include "cuptr.hpp"

#define BLK_Y 16
#define BLK_X 32

#define M 1024
#define N 2048

float A[M][N];

void assignData(){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            A[i][j] = i * N + j;
        }
    }
}

void valid(float *trans_arr_h){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            assert(abs(A[i][j]-trans_arr_h[j*M+i]) < 0.01);
        }
    }
}

__global__ void Transpose(float *arr, float *trans_arr){
    __shared__ float smem[BLK_Y][BLK_X+1];
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    int ldg_x = bix * bdx + tix;
    int ldg_y = biy * bdy + tiy;
    int idx = tiy * bdx + tix; 
    int lds_y = idx % BLK_Y;
    int lds_x = idx / BLK_Y;
    int stg_x = biy * bdy + lds_y;
    int stg_y = bix * bdx + lds_x;

    smem[tiy][tix] = arr[ldg_y * N + ldg_x];
    __syncthreads();
    trans_arr[stg_y * M + stg_x] = smem[lds_y][lds_x];
}

int main(){
    assignData();
    dim3 BLK_SIZE(BLK_X, BLK_Y);
    dim3 GRD_SIZE(N/BLK_X, M/BLK_Y);
    CuPtr<float> arr_d(M*N);
    checkCudaErrors(cudaMemcpy(arr_d.GetPtr(), A, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CuPtr<float> trans_arr_d(M*N);
    Transpose<<<GRD_SIZE, BLK_SIZE>>>(arr_d.GetPtr(), trans_arr_d.GetPtr());
    cudaDeviceSynchronize();
    HostPtr<float> trans_arr_h;
    trans_arr_d.ToHostPtr(trans_arr_h);
    valid(trans_arr_h.GetPtr());
}

// nvcc -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/trans Transpose.cu