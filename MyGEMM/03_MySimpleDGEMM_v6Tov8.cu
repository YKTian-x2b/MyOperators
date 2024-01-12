#include "../Common.cuh"
#include "../Utils/atomicAdd.cu"
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"
#include <string>

// transpose操作 要求M/N/K都至少是16
// #define M 512
// #define N 512
// #define K 2048
#define M 256
#define N 256
#define K 512
// M/N/K要大于等于SMEM_PART
#define SMEM_PART 32

#define T_BLOCK_X 16
#define T_BLOCK_Y 16

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
            // A[i][j] = u(e);
            // A[i][j] = (i*2+j)*0.001; 
            A[i][j] = 1;   
            // std::cout << A[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl; 
    // std::cout << "Matrix B: " << std::endl;
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            // B[i][j] = u(e);
            // B[i][j] = (i*2+j)*0.001;   
            B[i][j] = 1; 
            // std::cout << B[i][j] << " ";
        }
        // std::cout << std::endl;
    }
    // std::cout << std::endl;  
    memset(C, 0, sizeof(double) * M * N);
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

__global__ void DGEMM_v6(double * A, double * B, double * C){
    // __shared__ double A_part[SMEM_PART][SMEM_PART], B_part[SMEM_PART][SMEM_PART];
    // __shared__ double C_part[SMEM_PART][N];
    // // 线程组索引
    // int tix = threadIdx.x, tiy = threadIdx.y;
    // int bix = blockIdx.x, biy = blockIdx.y;
    // int bdx = blockDim.x, bdy = blockDim.y;
    // int gdx = gridDim.x, gdy = gridDim.y;

    // int thread_idx_x = bix * bdx + tix;
    // int thread_idx_y = biy * bdy + tiy;
    // int thread_x_len = gdx * bdx;
    // int thread_idx = thread_idx_y * thread_x_len + thread_idx_x;
    
    // int num_loops = N / SMEM_PART;

    // A_part[tiy][tix] = A[thread_idx];
    // for(int i = 0; i < num_loops; i++){
    //     C_part[tiy][tix + i * SMEM_PART] = 0.0;
    // }
    // __syncthreads();
    // for(int i = 0; i < num_loops; i++){
    //     B_part[tiy][tix] = B[(i * SMEM_PART + tiy) * K + bix * SMEM_PART + tix];
    //     __syncthreads();
    //     for(int j = 0; j < SMEM_PART; j++){
    //         atomicAdd_double(&C_part[tiy][i * SMEM_PART + (tiy + j) % SMEM_PART], A_part[tiy][tix] * B_part[(tiy + j) % SMEM_PART][tix]);
    //     }
    // }
    
    // for(int i = 0; i < num_loops; i++){
    //     atomicAdd_double(&C[biy * SMEM_PART * N + tiy * N + i * SMEM_PART + tix], C_part[tiy][i * SMEM_PART + tix]);
    // }
}

__global__ void DGEMM_v7(double * A, double * B, double * C){
    __shared__ double A_part[SMEM_PART][SMEM_PART], B_part[SMEM_PART][SMEM_PART];
    __shared__ double C_part[SMEM_PART][SMEM_PART];
    // 线程组索引
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    int num_loops = K / SMEM_PART;

    C_part[tiy][tix] = 0.0;
    for(int i = 0; i < num_loops; i++){
        int A_glo_x = i * SMEM_PART + tix;
        int A_glo_y = biy * SMEM_PART + tiy;
        int B_glo_x = A_glo_x;
        int B_glo_y = bix * SMEM_PART + tiy;
        A_part[tiy][tix] = A[A_glo_y * K + A_glo_x];
        B_part[tiy][tix] = B[B_glo_y * K + B_glo_x];
        __syncthreads();
        for(int j = 0; j < SMEM_PART; j++){
            atomicAdd_double(&C_part[tiy][(tiy + j) % SMEM_PART], A_part[tiy][tix] * B_part[(tiy + j) % SMEM_PART][tix]);
        }
    }
    int C_glo_x = bix * bdy + tix;
    int C_glo_y = biy * bdx + tiy;
    C[C_glo_y * N + C_glo_x] = C_part[tiy][tix];
}

__inline__ __device__ double warpReduce(double localSum){
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);
    return localSum;
}
// v8是失败的 因为寄存器数量不够
__global__ void DGEMM_v8(double * A, double * B, double * C){
    __shared__ double A_part[SMEM_PART][SMEM_PART], B_part[SMEM_PART][SMEM_PART];
    // __shared__ double C_part[SMEM_PART][SMEM_PART];
    volatile double C_part_reg[SMEM_PART];
    // 线程组索引
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    int num_loops = K / SMEM_PART;

    // C_part[tiy][tix] = 0.0;
    for(int i = 0; i < num_loops; i++){
        int A_glo_x = i * SMEM_PART + tix;
        int A_glo_y = biy * SMEM_PART + tiy;
        int B_glo_x = A_glo_x;
        int B_glo_y = bix * SMEM_PART + tiy;
        A_part[tiy][tix] = A[A_glo_y * K + A_glo_x];
        B_part[tiy][tix] = B[B_glo_y * K + B_glo_x];
        __syncthreads();
        for(int j = 0; j < SMEM_PART; j++){
            C_part_reg[(tiy + j) % SMEM_PART] += A_part[tiy][tix] * B_part[(tiy + j) % SMEM_PART][tix];
            // atomicAdd_double(&C_part[tiy][(tiy + j) % SMEM_PART], A_part[tiy][tix] * B_part[(tiy + j) % SMEM_PART][tix]);
        }
    }
    for(int i = 0; i < SMEM_PART; i++){
        int C_glo_x = bix * bdy + i;
        int C_glo_y = biy * bdx + tiy;
        atomicAdd_double(&C[C_glo_y * N + C_glo_x], C_part_reg[i]);
        // atomicAdd_double(&C_part[tiy][i], C_part_reg[i]);
        // C_part[tiy][i] += C_part_reg[i];
    }
    
    // // 改reduce
    // for(int i = 0; i < SMEM_PART; i++){
    //     C_part_reg[i] = warpReduce(C_part_reg[i]);
    // }
    // if(tix == 0){
    //     for(int i = 0; i < SMEM_PART; i++){
    //         C_part[tiy][i] = C_part_reg[i];
    //     }
    // }
    
    int C_glo_x = bix * bdy + tix;
    int C_glo_y = biy * bdx + tiy;
    // C[C_glo_y * N + C_glo_x] = C_part[tiy][tix];
}


__host__ void callDGEMM(double * d_A, double * d_B_T, double * d_C, double * d_B){
    int kernelIdx = 7;
    size_t C_Bytes = sizeof(double) * M * N;
    dim3 blockSize(1);
    dim3 gridSize(1);
    switch(kernelIdx){
        case 6:
            blockSize = {SMEM_PART, SMEM_PART};
            gridSize = {K/SMEM_PART, M/SMEM_PART};
            // printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            // printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            DGEMM_v6<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        case 7:
            blockSize = {SMEM_PART, SMEM_PART};
            gridSize = {N/SMEM_PART, M/SMEM_PART};
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            DGEMM_v7<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        case 8:
            blockSize = {SMEM_PART, SMEM_PART};
            gridSize = {N/SMEM_PART, M/SMEM_PART};
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            DGEMM_v8<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
    printf("DGEMM after: %s\n", cudaGetErrorString(cudaGetLastError()));
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

    return 0;
}