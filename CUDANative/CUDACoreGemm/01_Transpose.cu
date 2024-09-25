#include "../Common.cuh"
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"
#include <nvToolsExt.h>

// #define M 256
// #define N 256
// #define K 512
// #define T_BLOCK_X 16

#define M 8192
#define N 8192
#define K 16384
#define T_BLOCK_X 16
#define DGEMM_BLOCK 256

const double alpha = 1.0;
const double beta = 0.0;

double B[K][N];

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<double> u(1.5, 4.5);
    // std::cout << "Matrix B: " << std::endl;
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i][j] = u(e);
            // std::cout << B[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;
}

__host__ void printMatrix(double * matrix, const int rows, const int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            std::cout << matrix[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;  
}

__host__ void checkTransposeRes(double * B_T, double * B_T_blas, const int B_size){
    bool wrong = false;
    for(int i = 0; i < B_size; i++){
        double err = B_T[i] - B_T_blas[i];
        if(fabs(err) >=  0.01){
            wrong = true;
           // std::cout << "err: " << B_T[i] << " - " << B_T_blas[i] << " = " << fabs(err) << std::endl;
        }
    }
    if(wrong == false){
        std::cout << "transpose success!" << std::endl;
    }
    else{
        std::cout << "transpose error!" << std::endl;
    }
}

__global__ void transpose(double * B, double * B_T){
    __shared__ double smem[T_BLOCK_X][T_BLOCK_X+1];
    // 共享内存和线程块的结构是一样的，全局内存和网格的结构使用一样的。

    // 线程组索引
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    int gdx = gridDim.x, gdy = gridDim.y;

    int thread_x_len = gdx * bdx;
    int thread_y_len = gdy * bdy;

    // 既是线程索引也是全局内存B的索引
    int thread_idx_x = bix * bdx + tix;
    int thread_idx_y = biy * bdy + tiy;
    int thread_idx = thread_idx_y * thread_x_len + thread_idx_x;
    // 全局内存B_T的索引
    int trans_thread_idx_x = biy * bdy + tix;
    int trans_thread_idx_y = bix * bdx + tiy;
    int trans_thread_idx = trans_thread_idx_y * thread_y_len + trans_thread_idx_x;

    // 行读列写
    smem[tiy][tix] = B[thread_idx];
    __syncthreads();
    B_T[trans_thread_idx] = smem[tix][tiy];
}

__host__ void callTranspose(double * B, double * B_T, const int B_size){
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 自定义转置
    size_t B_Bytes = sizeof(double) * B_size;
    dim3 blockSize(T_BLOCK_X, T_BLOCK_X);
    dim3 gridSize(N/T_BLOCK_X, K/T_BLOCK_X);

    nvtxRangePushA("My");
    cudaEventRecord(start, 0);

    transpose<<<gridSize, blockSize>>>(B, B_T);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "MyTranspose cost " << elapsedTime  << " ms." << std::endl;
    // cudaDeviceSynchronize();
    nvtxRangePop();
    
    double * h_B_T = (double*)malloc(B_Bytes);
    CHECK(cudaMemcpy(h_B_T, B_T, B_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "h_B_T: " << std::endl;
    // printMatrix(h_B_T, N, K);
    
    // cublas转置
    double * B_T_blas;
    CHECK(cudaMalloc(&B_T_blas, B_Bytes));
    cublasHandle_t handle;
    cublasCreate(&handle);

    nvtxRangePushA("cublas");
    cudaEventRecord(start, 0);

    cublasStatus_t stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, K, N, &alpha, B, N, &beta, nullptr, N, B_T_blas, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "CublasTranspose cost " << elapsedTime  << " ms." << std::endl;
    //cudaDeviceSynchronize();
    nvtxRangePop();

    double * h_B_T_blas = (double*)malloc(B_Bytes);
    CHECK(cudaMemcpy(h_B_T_blas, B_T_blas, B_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "h_B_T_blas: " << std::endl;
    // printMatrix(h_B_T_blas, N,  K);
 
    // 正确性检查
    checkTransposeRes(h_B_T, h_B_T_blas, B_size);
    
    CHECK(cudaFree(B_T_blas));
    cublasDestroy(handle);
    free(h_B_T_blas);
    free(h_B_T);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    warmup();
    assignData();

    int B_size = K * N;
    double * d_B, * d_B_T;
    size_t B_Bytes = sizeof(double) * B_size;

    CHECK(cudaMalloc(&d_B, B_Bytes));
    CHECK(cudaMalloc(&d_B_T, B_Bytes));
    CHECK(cudaMemcpy(d_B, B, B_Bytes, cudaMemcpyHostToDevice));
    
    callTranspose(d_B, d_B_T, B_size);

    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_B_T));

    return 0;
}