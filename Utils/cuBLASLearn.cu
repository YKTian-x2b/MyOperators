#include "Common.cuh"

#include "cublas_v2.h"

#include <iostream>
#include <random>
#include <ctime>

// // C = aAB+bC; A:[m, k]; B:[k, n]; Operation表示是否需要转置
// cublasStatus_t cublasDgemm(cublasHandle handle, cublasOperation_t transa, cublasOperation_t transb, 
//                             int m, int n, int k, const double * alpha, const double * A, int lda,
//                             const double * B, int ldb, const float * beta, float * C, int ldc);
#define M 16
#define N 4
#define K 8

// #define M 256
// #define N 256
// #define K 512

const double alpha = 1.0;
const double beta = 0.0;

double A[M][K], B[K][N];
double C[M][N];

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<double> u(1.5, 4.5);
    std::cout << "Matrix A: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            // A[i][j] = u(e);
            A[i][j] = i*2+j;
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Matrix B: " << std::endl;
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            // B[i][j] = u(e);
            B[i][j] = i*2+j;
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    memset(C, 0, sizeof(double) * M * N);
}

void printRes(){
    std::cout << "Matrix C: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    assignData();

    double * d_A, * d_B, * d_C;
    size_t A_Bytes = sizeof(double) * M * K;
    size_t B_Bytes = sizeof(double) * K * N;
    size_t C_Bytes = sizeof(double) * M * N;
    CHECK(cudaMalloc(&d_A, A_Bytes));
    CHECK(cudaMalloc(&d_B, B_Bytes));
    CHECK(cudaMalloc(&d_C, C_Bytes));

    // cublasSetMatrix(M, K, sizeof(double), A, M, d_A, M);
    // cublasSetMatrix(K, N, sizeof(double), B, K, d_B, K);
    CHECK(cudaMemcpy(d_A, A, A_Bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, B_Bytes, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    // cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // 这样得到的结果是列存的C
    CHECK(cudaMemcpy(C, d_C, C_Bytes, cudaMemcpyDeviceToHost));
    // cublasGetMatrix(M, N, sizeof(double), d_C, M, C, M);

    printRes();

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    cublasDestroy(handle);
    return 0;
}

// nvcc -o cuBLASLearn cuBLASLearn.cu -l cublas

