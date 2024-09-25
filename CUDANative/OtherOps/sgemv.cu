#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"

/// 未加入双缓冲技术的结果 cublas：17.41ms； 自定义sgemm：18.21ms；

#define M 1
#define K 1024
#define N 1024
float A[M][K];
float B[K][N];
float C[N];
const float alpha = 1.0;
const float beta = 0.0;

#define BLK_SIZE 256
// 循环B矩阵的高维 即 K
#define FRAG_X 32
// 每个线程块处理 8*32个元素
#define FRAG_Y 8

#define FETCH_FLOAT4(ele) (reinterpret_cast<float4*>(&(ele))[0])

#define CHECK(call)                                                                  \
{                                                                                    \
    const cudaError_t error = call;                                                  \
    if(error != cudaSuccess){                                                        \
        printf("Error: %s: %d, ", __FILE__, __LINE__);                               \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));          \
        exit(1);                                                                     \
    }                                                                                \
}


__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(1.5, 4.5);

    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < K; j++){          
    #ifdef VALID
            A[i][j] = u(e);
    #else
            A[i][j] = u(e);
    #endif  
        }
    }

    for(size_t i = 0; i < K; i++){
        for(size_t j = 0; j < N; j++){
    #ifdef VALID
            B[i][j] = u(e);
    #else
            B[i][j] = u(e);
    #endif  
        }
    }

    memset(C, 0, sizeof(float) * M * N);
}

// 与方阵*方阵的优化区别在于：
// 没法将内积转换为外积 提高计算密度
// float4会降低并行度
__global__ void sgemm(float *A, float *B, float *C){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdx = blockDim.x;
    int warp_id = tid >> 5;

    __shared__ float smem[BLK_SIZE];

    float B_reg;
    float A_reg;
    float C_reg = 0.0;

    // 外层循环
    int num_loops_shared = K/FRAG_Y;
    // 这里需要做双缓冲优化，时间原因，没有改动。
    for(int i = 0; i < num_loops_shared; i++){
        // B gloMem-> reg
        // gloMem-> reg 8个warp分别读取8行 每行读32个float
        int B_ldg_x = bid * FRAG_X + (tid % 32);
        int B_ldg_y = i * FRAG_Y + (tid / 32);
        B_reg = B[B_ldg_y * N + B_ldg_x];
        // A gloMem-> reg
        A_reg = A[i * FRAG_Y + warp_id];
        
        // 计算
        C_reg += B_reg * A_reg;
    }

    // 为了避免全局内存写冲突 借用共享内存先reduce
    smem[tid] = C_reg;
    __syncthreads();
    
    if(tid < 32){
        float tmp_sum = C_reg;
        for(int i = 1; i < FRAG_Y; i++){
            tmp_sum += smem[tid + i*FRAG_X];
        }
        int C_stg_x = bid * FRAG_X + (tid % 32);
        C[C_stg_x] = tmp_sum;
    }
}


__host__ void callSGEMM(float * d_A, float * d_B, float * d_C){
    size_t C_Bytes = sizeof(float) * M * N;

    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float * d_C_cublas, * h_C_cublas;
    CHECK(cudaMalloc(&d_C_cublas, C_Bytes));
    h_C_cublas = (float *)malloc(C_Bytes);
    // cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, C_Bytes, cudaMemcpyDeviceToHost));

    // sgemm
    dim3 blockSize = {BLK_SIZE};
    // B矩阵的低维（N）做BLOCK级并行
    dim3 gridSize = {N/FRAG_X};
    printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
    printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
    sgemm<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    printf("SGEMM after: %s\n", cudaGetErrorString(cudaGetLastError()));
    float * h_C = (float *)malloc(C_Bytes);
    CHECK(cudaMemcpy(h_C, d_C, C_Bytes, cudaMemcpyDeviceToHost));

    // checkRes
    size_t err_cnt = 0;
    bool wrong = false;
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < N; j++){
            float err = h_C[i*N+j] - h_C_cublas[i*N+j];
            if(fabs(err) >=  0.1){
                err_cnt++;
                wrong = true;
                // std::cout << "err: [" << i << ", "  << j << "] " << h_C[i*N+j]  << " - " << h_C_cublas[i*N+j] << " = " << err << std::endl;
    #ifdef VALID
                std::cout << "err: [" << i << ", "  << j << "] " << h_C[i*N+j]  << " - " << h_C_cublas[i*N+j] << " = " << err << std::endl;
    #endif
            }
        }
    }
    if(!wrong){
        std::cout << "gemm success!" << std::endl;
    }
    else{
        std::cout << "gemm error! nums: " << err_cnt << std::endl;
    }
    
    cublasDestroy(handle);
    CHECK(cudaFree(d_C_cublas));
    free(h_C_cublas);
    free(h_C);
}


int main(){
    assignData();

    float *d_A, *d_B, *d_C;
    size_t A_Bytes = sizeof(float) * M * K;
    size_t B_Bytes = sizeof(float) * N * K;
    size_t C_Bytes = sizeof(float) * M * N;
    CHECK(cudaMalloc(&d_A, A_Bytes));
    CHECK(cudaMalloc(&d_B, B_Bytes));
    CHECK(cudaMalloc(&d_C, C_Bytes));
    CHECK(cudaMemcpy(d_A, A, A_Bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, B_Bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_C, 0, C_Bytes));

    callSGEMM(d_A, d_B, d_C);

    CHECK(cudaFree(d_A)); 
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    return 0;
}


// -DVALID 
// nvcc -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/sgemm sgemm.cu