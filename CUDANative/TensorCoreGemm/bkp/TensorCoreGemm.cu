#include <mma.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

#include <stdio.h>

#include "cuptr.hpp"
#include "hostptr.hpp"
#include "helper_cuda.h"

#define warpSize 32
const float alpha = 1.1f;
const float beta = 1.2f;

#define M 16
#define N 16
#define K 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256
#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

using namespace nvcuda;

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(1.5, 4.5);
    // std::cout << "Matrix A: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){          
    #ifdef VALID
            // A[i][j] = 1; 
            // A[i][j] = (i*2+j)*0.001; 
            A[i][j] = u(e);
    #else
            // A[i][j] = (i*2+j)*0.001;
            A[i][j] = u(e);
    #endif  
            // std::cout << A[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl; 
    // std::cout << "Matrix B: " << std::endl;
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
    #ifdef VALID
            // B[i][j] = 1; 
            // B[i][j] = (i*2+j)*0.001;
            B[i][j] = u(e);
    #else
            // B[i][j] = (i*2+j)*0.001;
            B[i][j] = u(e);
    #endif  
            // std::cout << B[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;  
    memset(C, 0, sizeof(float) * M * N);
}

// 一个tensorcore一个时钟周期可以做一个 D=A*B+C 的MMA操作。 其中 A/B/C/D都是4*4矩阵 A/B是FP16，C/D是FP32。
// 执行MMA操作的操作数必须在寄存器里，MMA是warp-level的，所以warp的每个线程都持有矩阵的一个fragment
// 矩阵参数和其fragment间的映射是不透明的，不应对此作出假设
// 仅作为WMMA API演示的GEMM， 没用到SMem
// C = alpha*A*B + beta*C
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha, float beta){
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 包含哪个矩阵，整个WMMA操作的形状，数据类型，以及矩阵的存储方式
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // 每个warp计算结果矩阵的一个tile
    for(int i = 0; i < k_ld; i += WMMA_K){
        int aCol = i;
        int bRow = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * N;
        if(aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld){
            // 从全局内存加载数据到一个fragment
            wmma::load_matrix_sync(a_frag, a + aCol + aRow*lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol*ldb, ldb);
            // mm
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;
    if(cRow < m_ld && cCol < n_ld){
        wmma::load_matrix_sync(c_frag, c + cCol + cRow*ldc, ldc, wmma::mem_row_major);
        for(int i = 0; i < c_frag.num_elements; i++){
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}


int main(){
    HostPtr<half> A_h(M_GLOBAL*K_GLOBAL);
    HostPtr<half> B_h(K_GLOBAL*N_GLOBAL);
    HostPtr<float> C_h(M_GLOBAL*N_GLOBAL);

    assignData();

    CuPtr<half> A_d(A_h);
    CuPtr<half> B_d(B_h);
    CuPtr<float> C_d(C_h);
    CuPtr<float> D_d(M_GLOBAL*N_GLOBAL);
    checkCudaErrors(cudaMemset(D_d, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
    

    dim3 blockDim(128, 4, 1);
    dim3 gridDim;
    int num_GD_X_WMMA = (blockDim.x / 32) * WMMA_M;
    int num_GD_Y_WMMA = blockDim.y * WMMA_N; 
    gridDim.x = (M_GLOBAL + num_GD_X_WMMA - 1) / num_GD_X_WMMA;
    gridDim.y = (N_GLOBAL + num_GD_Y_WMMA - 1) / num_GD_Y_WMMA;
    simple_wmma_gemm<<<gridDim, blockDim>>>(A_d.GetPtr(), B_d.GetPtr(), C_d.GetPtr(), D_d.GetPtr(), M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
}

// nvcc -gencode=arch=compute_86,code=\"sm_86,compute_86\" -I../Utils -L /usr/local/cuda/lib64 -l cuda -o TensorCoreGemm TensorCoreGemm.cu