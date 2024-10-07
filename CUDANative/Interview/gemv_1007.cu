#include "../common.h"
#include <cublas_v2.h>

#define Dtype float

#define FLOAT4(elt) (*reinterpret_cast<float4*>(&(elt)))

template<typename T, unsigned N, unsigned THDS_PER_KEY>
inline __device__ float dot(const float4 (&q)[N], const float4 (&cacheK)[N]){
    float4 accum = make_float4(0., 0., 0., 0.);
    for(int i = 0; i < N; i++){
        accum.x += q[i].x * cacheK[i].x;
        accum.y += q[i].y * cacheK[i].y;
        accum.z += q[i].z * cacheK[i].z;
        accum.w += q[i].w * cacheK[i].w;
    }
    float accum_final = accum.x + accum.y + accum.z + accum.w;

    #pragma unroll
    for (int mask = THDS_PER_KEY / 2; mask >= 1; mask /= 2) {
        accum_final += __shfl_xor_sync(uint32_t(-1), accum_final, mask);
    }
    return accum_final;
}


template<typename T, unsigned BLK_SIZE, unsigned HEAD_DIM, unsigned THDS_PER_KEY, unsigned UNROLL=4>
__global__ void sgemv(T* Q, T* cacheK, T* S, const unsigned SEQ_LEN){
    // Q [dim] cacheK [seq_len, head_dim]
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    constexpr unsigned K_VEC_SIZE = 16 / sizeof(T);
    constexpr unsigned KEYS_PER_ITER = BLK_SIZE / THDS_PER_KEY;
    constexpr unsigned K_VECS_PER_KEY = HEAD_DIM / (THDS_PER_KEY * K_VEC_SIZE);

    int cacheK_time_base = bid * KEYS_PER_ITER * UNROLL + tid / THDS_PER_KEY;
    int chunk_id_base = (tid % THDS_PER_KEY) * K_VEC_SIZE;

    float4 q_vecs[K_VECS_PER_KEY];
    float4 cacheK_vecs[UNROLL][K_VECS_PER_KEY];
    
    #pragma unroll
    for(int j = 0; j < K_VECS_PER_KEY; j++){
        int q_chunk_id = chunk_id_base + j * THDS_PER_KEY * K_VEC_SIZE;
        q_vecs[j] = FLOAT4(Q[q_chunk_id]);
    }

    #pragma unroll
    for(int i = 0; i < UNROLL; i++){
        int cacheK_time_nw = cacheK_time_base + i * KEYS_PER_ITER;
        #pragma unroll
        for(int j = 0; j < K_VECS_PER_KEY; j++){
            int cacheK_chunk_id = chunk_id_base + j * THDS_PER_KEY * K_VEC_SIZE;
            cacheK_vecs[i][j] = FLOAT4(cacheK[cacheK_time_nw * HEAD_DIM + cacheK_chunk_id]);
        }
    }

    #pragma unroll
    for(int i = 0; i < UNROLL; i++){
        int cacheK_time_nw = cacheK_time_base + i * KEYS_PER_ITER;
        S[cacheK_time_nw] = dot<T, K_VECS_PER_KEY, THDS_PER_KEY>(q_vecs, cacheK_vecs[i]);
    }
}


int main(){
    constexpr unsigned M{1024 * 8};//
    constexpr unsigned N{128 * 16}; // * 8

    constexpr unsigned BLK_SIZE{256};
    // [M, N] * [N] => [M]
    // [seq_len, dim] * [dim] = [seq_len]
    // Q [N] cacheK [M, N] 一个warp负责某个M所有的N
    constexpr unsigned threads_per_key_compute = N * sizeof(Dtype) / 16;
    constexpr unsigned THREADS_PER_KEY = 32 < threads_per_key_compute ? 32 : threads_per_key_compute;
    constexpr unsigned UNROLL = 4;
    constexpr unsigned KEYS_PER_BLK = BLK_SIZE / THREADS_PER_KEY * UNROLL;

    constexpr unsigned GRD_SIZE{(M-1+KEYS_PER_BLK)/KEYS_PER_BLK};

    Dtype* d_Q;
    Dtype* d_cacheK;
    GpuAllocateMatrix<Dtype>(&d_Q, 1, N, 1);
    GpuAllocateMatrix<Dtype>(&d_cacheK, M, N, 1);

    // Dtype* h_cacheK;
    // size_t size_cacheK = sizeof(Dtype) * M * N;
    // CUDA_CHECK(cudaMallocHost((void **)&h_cacheK, size_cacheK));
    // CUDA_CHECK(cudaMemcpy(h_cacheK, d_cacheK, size_cacheK, cudaMemcpyDeviceToHost));
    // PrintMatrix<Dtype>(h_cacheK, M, N);


    Dtype* d_S;
    GpuAllocateMatrix<Dtype>(&d_S, 1, M, 0);

    sgemv<Dtype, BLK_SIZE, N, THREADS_PER_KEY, UNROLL><<<GRD_SIZE, BLK_SIZE>>>(d_Q, d_cacheK, d_S, M);
    cudaDeviceSynchronize();

    Dtype* h_S;
    size_t size_S = sizeof(Dtype) * M;
    CUDA_CHECK(cudaMallocHost((void **)&h_S, size_S));
    CUDA_CHECK(cudaMemcpy(h_S, d_S, size_S, cudaMemcpyDeviceToHost));


    /// cublas sgemv
    Dtype* d_S_cublas;
    GpuAllocateMatrix<Dtype>(&d_S_cublas, 1, M, 0);
    static_assert(sizeof(Dtype) == sizeof(float));

    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
    //     int m, int n,
    //     const float           *alpha,
    //     const float           *A, int lda,
    //     const float           *x, int incx,
    //     const float           *beta,
    //     float           *y, int incy)
    cublasSgemv (blas_handle, CUBLAS_OP_T, 
        N, M, 
        &alpha, 
        d_cacheK, N, 
        d_Q, 1, 
        &beta, d_S_cublas, 1
    );

    Dtype* h_S_cublas;
    CUDA_CHECK(cudaMallocHost((void **)&h_S_cublas, size_S));
    CUDA_CHECK(cudaMemcpy(h_S_cublas, d_S_cublas, size_S, cudaMemcpyDeviceToHost));

    CheckRes(h_S, h_S_cublas, 1, M);


    cublasDestroy(blas_handle);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_cacheK));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_S_cublas));
    CUDA_CHECK(cudaFreeHost(h_S));
    CUDA_CHECK(cudaFreeHost(h_S_cublas));
}

// nvcc -I/home/yujixuan/kaiPro/GitHubLocal/MyOperators/CUDANative -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/gemv gemv_1007.cu
// ncu -f -o midRes/gemv_1007_prof --set full res/gemv