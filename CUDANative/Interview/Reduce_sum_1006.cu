#include <cuda_runtime.h>
#include "../common.h"

#define Dtype float

__forceinline__ __device__ float warpReduce(float elt){
    elt += __shfl_xor_sync(unsigned(-1), elt, 16);
    elt += __shfl_xor_sync(unsigned(-1), elt, 8);
    elt += __shfl_xor_sync(unsigned(-1), elt, 4);
    elt += __shfl_xor_sync(unsigned(-1), elt, 2);
    elt += __shfl_xor_sync(unsigned(-1), elt, 1);
    return elt;
}


template<typename T,
         unsigned N_DATA,
         unsigned UNROLL = 8>
__global__ void Reduce_sum(T *in, T *out){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdx = blockDim.x;

    int idx = bid * bdx * UNROLL + tid;

    extern __shared__ float smem[];

    float sum = 0.;
    #pragma unroll
    for(int i = idx, j = 0; i < N_DATA, j < UNROLL; i += bdx, j++){
        sum += in[i];
    }

    smem[tid] = sum;
    __syncthreads();


    if(bdx >= 1024 && tid < 512){
        smem[tid] += smem[tid + 512];
    }
    __syncthreads();
    if(bdx >= 512 && tid < 256){
        smem[tid] += smem[tid + 256];
    }
    __syncthreads();
    if(bdx >= 256 && tid < 128){
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();
    if(bdx >= 128 && tid < 64){
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();
    if(bdx >= 64 && tid < 32){
        smem[tid] += smem[tid + 32];
    }
    __syncthreads();

    if(tid < 32){
        smem[tid] = warpReduce(smem[tid] );
    }

    if(tid == 0){
        out[bid] = smem[0];
    } 
}

template<typename T>
float cpu_reduce_sum(T *in, const unsigned N_DATA){
    float sum = 0.;
    for(int i = 0; i < N_DATA; i++){
        sum += in[i];
    }
    return sum;
}

int main(){
    constexpr unsigned N_DATA{1024*2};
    constexpr unsigned BLK_SIZE{256};
    constexpr unsigned GRD_SIZE{(N_DATA-1+BLK_SIZE)/BLK_SIZE};
    
    Dtype* h_out;
    CpuAllocateMatrix<Dtype>(&h_out, 1, GRD_SIZE, 0);

    Dtype* d_in, *d_out;
    GpuAllocateMatrix<Dtype>(&d_in, 1, N_DATA, 2);
    GpuAllocateMatrix<Dtype>(&d_out, 1, GRD_SIZE, 0);

    constexpr unsigned smem_size_bytes{sizeof(float) * BLK_SIZE};
    Reduce_sum<Dtype, N_DATA, 8><<<GRD_SIZE, BLK_SIZE, smem_size_bytes, 0>>>(d_in, d_out);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(Dtype)*GRD_SIZE, cudaMemcpyDeviceToHost));

    float gpu_sum_final = 0.;
    for(int i = 0; i < GRD_SIZE; i++){
        gpu_sum_final += h_out[i];
    }
    std::cout << "gpu_sum_final: " << gpu_sum_final << std::endl;


    Dtype* h_in;
    size_t size_in = sizeof(Dtype) * N_DATA;
    CUDA_CHECK(cudaMallocHost((void **)&h_in, size_in));
    CUDA_CHECK(cudaMemcpy(h_in, d_in, size_in, cudaMemcpyDeviceToHost));
    float cpu_sum_final = cpu_reduce_sum<Dtype>(h_in, N_DATA);
    std::cout << "cpu_sum_final: " << cpu_sum_final << std::endl;

    // 释放
}


// nvcc -I/home/yujixuan/kaiPro/GitHubLocal/MyOperators/CUDANative -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/reduce_sum Reduce_sum_1006.cu