#include "../Common.cuh"
#include "../Utils/atomicAdd.cu"
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"
#include <string>

// transpose操作 要求M/N/K都至少是16
// 共享内存分配决定了一个SM能启动的Block数，所以应该限制SMEM_BOUND 使 （maxWarpPerSM/WarpPerBlock）* SMEM_BOUND < SMEMPerSM
#define SMEM_BOUND 1024
#define M 512
#define N 512
#define K 2048

#define T_BLOCK_X 16
#define T_BLOCK_Y 16

#define DGEMM_BLOCK 256

const double alpha = 1.0;
const double beta = 0.0;

double A[M][K], B[K][N];
double C[M][N];

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<double> u(1.5, 4.5);
    // std::cout << "Matrix A: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i][j] = u(e);
            // A[i][j] = (i*2+j)*0.001;    
            // std::cout << A[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl; 
    // std::cout << "Matrix B: " << std::endl;
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i][j] = u(e);
            // B[i][j] = (i*2+j)*0.001;    
            // std::cout << B[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;  
    memset(C, 0, sizeof(double) * M * N);
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

__host__ void HostCompute(){
    memset(C, 0, sizeof(double) * M * N);
    std::cout << "Matrix C: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
            printf("%lf ", C[i][j]);
        }
        printf("\n");
    }
}

__global__ void transpose(double * B, double * B_T){
    __shared__ double smem[T_BLOCK_Y][T_BLOCK_X];
    // 共享内存和线程块的结构是一样的，全局内存和网格的结构使用一样的。

    // 线程组索引
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    int gdx = gridDim.x, gdy = gridDim.y;

    int thread_idx_x = bix * bdx + tix;
    int thread_idx_y = biy * bdy + tiy;
    int thread_x_len = gdx * bdx;
    int thread_y_len = gdy * bdy;
    // 既是线程索引也是全局内存B的索引
    int thread_idx = thread_idx_y * thread_x_len + thread_idx_x;
    // B_T索引
    int tran_thread_idx = thread_idx_x * thread_y_len + thread_idx_y;

    // 行读列写
    smem[tiy][tix] = B[thread_idx];
    B_T[tran_thread_idx] = smem[tiy][tix];

}
// 只加了第一行
__global__ void DGEMM_v1(double * A, double * B, double * C){
    // 共享内存限制48KB 未考虑
    __shared__ double smem_B[K];
    __shared__ double smem_midRes[DGEMM_BLOCK];
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;

    int tix_tmp = tix;
    #pragma unroll
    while(tix_tmp < K){
        smem_B[tix_tmp] = B[bix * K + tix_tmp];
        smem_midRes[tix] += smem_B[tix_tmp] * A[tix_tmp];
        tix_tmp += bdx;
    }
    __syncthreads();

    if(bdx >= 1024 && tix < 512){
        smem_midRes[tix] += smem_midRes[tix+512];
    }
    __syncthreads();
    if(bdx >= 512 && tix < 256){
        smem_midRes[tix] += smem_midRes[tix+256];
    }
    __syncthreads();
    if(bdx >= 256 && tix < 128){
        smem_midRes[tix] += smem_midRes[tix+128];
    }
    __syncthreads();
    if(bdx >= 128 && tix < 64){
        smem_midRes[tix] += smem_midRes[tix+64];
    }
    __syncthreads();

    if(tix < 32){
        volatile double * smem_midRes_vol = smem_midRes;
        smem_midRes_vol[tix] += smem_midRes_vol[tix+32];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+16];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+8];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+4];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+2];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+1];
    }

    if(tix == 0){
        C[bix] = smem_midRes[tix];
    }
}

__global__ void DGEMM_v2(double * A, double * B, double * C){
    // 共享内存限制48KB 未考虑
    __shared__ double smem_midRes[DGEMM_BLOCK];
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int biy = blockIdx.y;
    int bdx = blockDim.x;
    int gdx = gridDim.x;

    int tix_tmp = tix;
    smem_midRes[tix] = 0.0;
    #pragma unroll
    while(tix_tmp < K){
        smem_midRes[tix] += B[bix * K + tix_tmp] * A[biy * K + tix_tmp];
        tix_tmp += bdx;
    }
    __syncthreads();

    if(bdx >= 1024 && tix < 512){
        smem_midRes[tix] += smem_midRes[tix+512];
    }
    __syncthreads();
    if(bdx >= 512 && tix < 256){
        smem_midRes[tix] += smem_midRes[tix+256];
    }
    __syncthreads();
    if(bdx >= 256 && tix < 128){
        smem_midRes[tix] += smem_midRes[tix+128];
    }
    __syncthreads();
    if(bdx >= 128 && tix < 64){
        smem_midRes[tix] += smem_midRes[tix+64];
    }
    __syncthreads();

    if(tix < 32){
        volatile double * smem_midRes_vol = smem_midRes;
        smem_midRes_vol[tix] += smem_midRes_vol[tix+32];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+16];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+8];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+4];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+2];
        smem_midRes_vol[tix] += smem_midRes_vol[tix+1];
    }
    if(tix == 0){
        C[biy * gdx + bix] = smem_midRes[tix];
    }
}
// 只能处理K==SMEM_BOUND的情况
// 如果K>SMEM_BOUND，那么需要考虑如何区分一个block对应的行偏移 
// 如果K<SMEM_BOUND，那么需要考虑如何区分一个block对应的多行 
__global__ void DGEMM_v3(double * A, double * B, double * C){
    // 以SMEM上限为单位 一个block处理一个单位 一次处理完 A数组该单位内 所有元素所需的计算
    // grid 处理完A数组的所有单位
    __shared__ double smem_midRes[SMEM_BOUND];
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    // 每个block需处理单位的首地址
    double * A_start = A + SMEM_BOUND * bix;
    // 先把单位内的所有元素读入共享内存
    int tix_tmp = tix;
    #pragma unroll
    while(tix_tmp < SMEM_BOUND){
        smem_midRes[tix_tmp] = A_start[tix_tmp];
        tix_tmp += bdx;
    }
    __syncthreads();
    // 外层循环遍历B数组 读取 A数组该单位内所有元素计算所需的对应B元素
    // 外层循环内就是标准的reduce
    #pragma unroll
    for(int i = 0; i < N; i++){
        __shared__ double smem_tmp[DGEMM_BLOCK];
        smem_tmp[tix] = 0.0;
        #pragma unroll
        for(int tix_tmp_mid = tix; tix_tmp_mid < SMEM_BOUND; tix_tmp_mid += bdx){
            smem_tmp[tix] += smem_midRes[tix_tmp_mid] * B[i * K + tix_tmp_mid];
        }
        __syncthreads();
        // smem_tmp[] -> C[]
        if(bdx >= 1024 && tix < 512){
            smem_tmp[tix] += smem_tmp[tix+512];
        }
        __syncthreads();
        if(bdx >= 512 && tix < 256){
            smem_tmp[tix] += smem_tmp[tix+256];
        }
        __syncthreads();
        if(bdx >= 256 && tix < 128){
            smem_tmp[tix] += smem_tmp[tix+128];
        }
        __syncthreads();
        if(bdx >= 128 && tix < 64){
            smem_tmp[tix] += smem_tmp[tix+64];
        }
        __syncthreads();
    
        if(tix < 32){
            volatile double * smem_tmp_vol = smem_tmp;
            smem_tmp_vol[tix] += smem_tmp_vol[tix+32];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+16];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+8];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+4];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+2];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+1];
        }
        if(tix == 0){
            int C_row = bix/(K/SMEM_BOUND);
            C[C_row * N + i] += smem_tmp[tix];
        }
    }
}

// K >= SMEM_BOUND
__global__ void DGEMM_v4_situ1(double * A, double * B, double * C){
    // 以SMEM上限为单位 一个block处理一个单位 一次处理完 A数组该单位内 所有元素所需的计算
    // grid 处理完A数组的所有单位
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    // 要注意除0错误
    int A_row = bix/(K/SMEM_BOUND);
    int A_offset = (bix % (K/SMEM_BOUND)) * SMEM_BOUND;
    __shared__ double smem_midRes[SMEM_BOUND];
    // 每个block需处理单位的首地址
    double * A_start = A + A_row * K + A_offset;
    // 先把单位内的所有元素读入共享内存
    int tix_tmp = tix;
    #pragma unroll
    while(tix_tmp < SMEM_BOUND){
        smem_midRes[tix_tmp] = A_start[tix_tmp];
        tix_tmp += bdx;
    }
    __syncthreads();
    // 外层循环遍历B数组 读取 A数组该单位内所有元素计算所需的对应B元素
    // 外层循环内就是标准的reduce
    __shared__ double smem_tmp[DGEMM_BLOCK];
    #pragma unroll
    for(int i = 0; i < N; i++){
        smem_tmp[tix] = 0.0;
        __syncthreads();
        #pragma unroll
        for(int tix_tmp_mid = tix; tix_tmp_mid < SMEM_BOUND; tix_tmp_mid += bdx){
            smem_tmp[tix] += smem_midRes[tix_tmp_mid] * B[i * K + A_offset + tix_tmp_mid];
        }
        __syncthreads();
        // smem_tmp[] -> C[]
        if(bdx >= 1024 && tix < 512){
            smem_tmp[tix] += smem_tmp[tix+512];
        }
        __syncthreads();
        if(bdx >= 512 && tix < 256){
            smem_tmp[tix] += smem_tmp[tix+256];
        }
        __syncthreads();
        if(bdx >= 256 && tix < 128){
            smem_tmp[tix] += smem_tmp[tix+128];
        }
        __syncthreads();
        if(bdx >= 128 && tix < 64){
            smem_tmp[tix] += smem_tmp[tix+64];
        }
        __syncthreads();
    
        if(tix < 32){
            volatile double * smem_tmp_vol = smem_tmp;
            smem_tmp_vol[tix] += smem_tmp_vol[tix+32];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+16];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+8];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+4];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+2];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+1];
        }
        if(tix == 0){
            atomicAdd_double(&C[A_row * N + i], smem_tmp[tix]);
        }
        __syncthreads();
    }
}
// K < SMEM_BOUND
__global__ void DGEMM_v4_situ2(double * A, double * B, double * C){
    // int tix = threadIdx.x;
    // int bix = blockIdx.x;
    // int bdx = blockDim.x;
    // int constexpr num_rows = SMEM_BOUND/K;
    // double * A_start = A + bix * num_rows * K;
    // __shared__ double smem_midRes[SMEM_BOUND];
    // // 先把单位内的所有元素读入共享内存
    // int tix_tmp = tix;
    // #pragma unroll
    // while(tix_tmp < SMEM_BOUND){
    //     smem_midRes[tix_tmp] = A_start[tix_tmp];
    //     tix_tmp += bdx;
    // }
    // __syncthreads();
    // // 外层循环遍历B数组 读取 A数组该单位内所有元素计算所需的对应B元素
    // __shared__ double smem_tmp[num_rows][DGEMM_BLOCK];
    // __shared__ double smem_tmp_B[K];
    // #pragma unroll
    // for(int i = 0; i < N; i++){
    //     #pragma unroll
    //     for(int j = 0; j < num_rows; j++)
    //         smem_tmp[j][tix] = 0.0;
    //     #pragma unroll
    //     for(int tix_tmp_mid = tix; tix_tmp_mid < K; tix_tmp_mid += bdx){
    //         // 全局内存冲突
    //         // smem_tmp_B[tix_tmp_mid] = B[i * K + tix_tmp_mid];
    //         smem_tmp_B[tix_tmp_mid] = B[((i+bix) % N) * K + tix_tmp_mid];
    //     }
    //     __syncthreads();
    //     // 内层循环遍历SMEM_BOUND覆盖的数组A的多行
    //     #pragma unroll
    //     for(int j = 0; j < num_rows; j++){
    //         #pragma unroll
    //         for(int tix_tmp_mid = tix; tix_tmp_mid < K; tix_tmp_mid += bdx){
    //             smem_tmp[j][tix] += smem_midRes[j * K + tix_tmp_mid] * smem_tmp_B[tix_tmp_mid];
    //         }
    //         __syncthreads();
    //         // smem_tmp[] -> C[]
    //         if(bdx >= 1024 && tix < 512){
    //             smem_tmp[j][tix] += smem_tmp[j][tix+512];
    //         }
    //         __syncthreads();
    //         if(bdx >= 512 && tix < 256){
    //             smem_tmp[j][tix] += smem_tmp[j][tix+256];
    //         }
    //         __syncthreads();
    //         if(bdx >= 256 && tix < 128){
    //             smem_tmp[j][tix] += smem_tmp[j][tix+128];
    //         }
    //         __syncthreads();
    //         if(bdx >= 128 && tix < 64){
    //             smem_tmp[j][tix] += smem_tmp[j][tix+64];
    //         }
    //         __syncthreads();
    //         if(tix < 32){
    //             volatile double * smem_tmp_vol = smem_tmp[j];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+32];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+16];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+8];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+4];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+2];
    //             smem_tmp_vol[tix] += smem_tmp_vol[tix+1];
    //         }
    //         if(tix == 0){
    //             atomicAdd_double(&C[(bix * num_rows + j) * N + ((i+bix) % N)], smem_tmp[j][tix]);
    //         }
    //         __syncthreads();
    //     }
    // }
}

// K < SMEM_BOUND
// 以K为单位 一个block处理一个单位 一次处理完 A数组该单位内 所有元素所需的计算
// grid 处理完A数组的所有单位
__global__ void DGEMM_v5_situ2(double * A, double * B, double * C){
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    double * A_start = A + bix * K;
    __shared__ double smem_midRes[K];
    // 先把单位内的所有元素读入共享内存
    #pragma unroll
    for(int tix_tmp = tix; tix_tmp < K; tix_tmp += bdx){
        smem_midRes[tix_tmp] = A_start[tix_tmp];
    }
    __syncthreads();
    __shared__ double smem_tmp[DGEMM_BLOCK];
    #pragma unroll
    for(int i = 0; i < N; i++){
        smem_tmp[tix] = 0.0;
        __syncthreads();
        #pragma unroll
        for(int tix_tmp_mid = tix; tix_tmp_mid < K; tix_tmp_mid += bdx){
            smem_tmp[tix] += smem_midRes[tix_tmp_mid] * B[((i+bix)%N)* K + tix_tmp_mid];
        }
        __syncthreads();
        // smem_tmp[] -> C[]
        if(bdx >= 1024 && tix < 512){
            smem_tmp[tix] += smem_tmp[tix+512];
        }
        __syncthreads();
        if(bdx >= 512 && tix < 256){
            smem_tmp[tix] += smem_tmp[tix+256];
        }
        __syncthreads();
        if(bdx >= 256 && tix < 128){
            smem_tmp[tix] += smem_tmp[tix+128];
        }
        __syncthreads();
        if(bdx >= 128 && tix < 64){
            smem_tmp[tix] += smem_tmp[tix+64];
        }
        __syncthreads();
    
        if(tix < 32){
            volatile double * smem_tmp_vol = smem_tmp;
            smem_tmp_vol[tix] += smem_tmp_vol[tix+32];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+16];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+8];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+4];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+2];
            smem_tmp_vol[tix] += smem_tmp_vol[tix+1];
        }
        if(tix == 0){
            atomicAdd_double(&C[bix * N + (i+bix)%N], smem_tmp[tix]);
        }
        __syncthreads();
    }
}

__host__ void callDGEMM_v4(double * d_A, double * d_B_T, double * d_C, dim3 gridSize, dim3 blockSize){
    // printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
    // printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
    if(K >= SMEM_BOUND){
        printf("DGEMM_v4_situ1\n");
        DGEMM_v4_situ1<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
    }
    else{
        printf("DGEMM_v4_situ2\n");
        DGEMM_v4_situ2<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
    }
}

__host__ void callDGEMM_v5(double * d_A, double * d_B_T, double * d_C, dim3 gridSize, dim3 blockSize){
    // printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
    // printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
    printf("DGEMM_v5_situ2\n");
    DGEMM_v5_situ2<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
}

__host__ void checkRes(double * B_T, double * B_T_blas, const int B_size){
    bool wrong = false;
    for(int i = 0; i < B_size; i++){
        double err = B_T[i] - B_T_blas[i];
        if(fabs(err) >=  0.01){
            wrong = true;
            std::cout << "err: " << B_T[i] << " - " << B_T_blas[i] << " = " << fabs(err) << std::endl;
        }
    }
    if(wrong == false){
        std::cout << "transpose success!" << std::endl;
    }
    else{
        std::cout << "transpose error!" << std::endl;
    }
}

// cublasStatus_t cublasDgeam(cublasHandle_t handle,
//     cublasOperation_t transa, cublasOperation_t transb,
//     int m, int n,
//     const double          *alpha,
//     const double          *A, int lda,
//     const double          *beta,
//     const double          *B, int ldb,
//     double          *C, int ldc)
__host__ void callTranspose(double * B, double * B_T, const int B_size){
    size_t B_Bytes = sizeof(double) * B_size;

    dim3 blockSize(T_BLOCK_X, T_BLOCK_Y);
    dim3 gridSize(N/T_BLOCK_X, K/T_BLOCK_Y);
    transpose<<<gridSize, blockSize>>>(B, B_T);
    cudaDeviceSynchronize();
    double * h_B_T = (double*)malloc(B_Bytes);
    CHECK(cudaMemcpy(h_B_T, B_T, B_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "h_B_T: " << std::endl;
    // printMatrix(h_B_T, N, K);
    
    
    double * B_T_blas;
    CHECK(cudaMalloc(&B_T_blas, B_Bytes));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, K, N, &alpha, B, N, &beta, nullptr, N, B_T_blas, K);
    cudaDeviceSynchronize();
    double * h_B_T_blas = (double*)malloc(B_Bytes);
    CHECK(cudaMemcpy(h_B_T_blas, B_T_blas, B_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "h_B_T_blas: " << std::endl;
    // printMatrix(h_B_T_blas, N,  K);
 
    checkRes(h_B_T, h_B_T_blas, B_size);

    free(h_B_T);
    CHECK(cudaFree(B_T_blas));
    cublasDestroy(handle);
    free(h_B_T_blas);
}

__host__ void callDGEMM(double * d_A, double * d_B_T, double * d_C, double * d_B){
    int kernelIdx = 4;
    size_t C_Bytes = sizeof(double) * M * N;
    dim3 blockSize(DGEMM_BLOCK);
    dim3 gridSize(N);
    switch(kernelIdx){
        case 1:
            DGEMM_v1<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        case 2:   
            gridSize = {N, M};
            DGEMM_v2<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        case 3:
            gridSize = M * K / SMEM_BOUND;
            DGEMM_v3<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        case 4:
            gridSize = M * K / SMEM_BOUND;
            callDGEMM_v4(d_A, d_B_T, d_C, gridSize, blockSize);
            break;
        case 5:
            // case5只考虑 K < SMEM_BOUND
            gridSize = M;
            callDGEMM_v5(d_A, d_B_T, d_C, gridSize, blockSize);
        default:
            break;
    }
    cudaDeviceSynchronize();
    printf("DGEMM_v4 after: %s\n", cudaGetErrorString(cudaGetLastError()));
    double * h_C = (double *)malloc(C_Bytes);
    CHECK(cudaMemcpy(h_C, d_C, C_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "Matrix C: " << std::endl;
    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++){
    //         std::cout << h_C[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    double * d_C_cublas, * h_C_cublas;
    CHECK(cudaMalloc(&d_C_cublas, C_Bytes));
    h_C_cublas = (double *)malloc(C_Bytes);
    // cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, C_Bytes, cudaMemcpyDeviceToHost));
    // std::cout << "Matrix C_cublas: " << std::endl;
    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++){
    //         std::cout << h_C_cublas[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // checkRes
    int err_cnt = 0;
    bool wrong = false;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            double err = h_C[i*N+j] - h_C_cublas[i*N+j];
            if(fabs(err) >=  0.001){
                err_cnt++;
                wrong = true;
                // std::cout << "err: [" << i << ", "  << j << "] " << h_C[i*N+j]  << " - " << h_C_cublas[i*N+j] << " = " << err << std::endl;
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


int main(int argc, char **argv){
    assignData();

    // GPU 计算
    int B_size = K * N;
    double * d_A, * d_B, * d_C, * d_B_T;
    size_t A_Bytes = sizeof(double) * M * K;
    size_t B_Bytes = sizeof(double) * B_size;
    size_t C_Bytes = sizeof(double) * M * N;
    CHECK(cudaMalloc(&d_A, A_Bytes));
    CHECK(cudaMalloc(&d_B, B_Bytes));
    CHECK(cudaMalloc(&d_C, C_Bytes));
    CHECK(cudaMalloc(&d_B_T, B_Bytes));
    CHECK(cudaMemcpy(d_A, A, A_Bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, B_Bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_C, 0, C_Bytes));
    
    callTranspose(d_B, d_B_T, B_size);
    callDGEMM(d_A, d_B_T, d_C, d_B);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_B_T));

    // CPU 计算
    // HostCompute();
    // printf("transpose before: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}