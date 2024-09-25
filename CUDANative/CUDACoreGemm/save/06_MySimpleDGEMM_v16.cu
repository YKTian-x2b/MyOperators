#include "../Common.cuh"
#include "../Utils/atomicAdd.cu"
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"
#include <string>

// regsPerBlock和regsPerSM都是65536 这些是指32bit的寄存器
// 如果启动 3个block; 一个block 占 256*64个寄存器 存256*32个double
// 如果启动 2个block；一个block 占 256*128个寄存器 存256*64个double
// 如果启动 1个block; 一个block 占 256*256个寄存器 存256*128个double

// 为了用满SM的3*16个warp，限制一个block是32*16个线程 一个SM启动三个block，那么一个block先用1/4regs 256*64个
// 如果是连续两个regs存一个double的话，一个block处理256*32个，就是16个double每线程?

// smemPerBlock是49152=128*128*3 smemPerSM是102400=32*32*100

// transpose操作 要求M/N/K都至少是16 T_BLOCK_X/T_BLOCK_Y
// gemm操作 要求M/N/K要大于等于GEMM_BLOCK_Y
// #define M 32
// #define N 32
// #define K 512

// #define M 512
// #define N 512
// #define K 2048
#define M 512
#define N 512
#define K 2048

// 转置用的blockDim 
#define T_BLOCK_X 16
#define T_BLOCK_Y 16

// 矩阵乘用的blockDim 
#define GEMM_BLOCK_X 32
#define GEMM_BLOCK_Y 8
// 用来reg_doublebuffering 一次global->shared填充能带来的 shared->reg填充数
#define REG_DB 4

// 共享内存矩阵的X维
#define SMEM_PART_X (GEMM_BLOCK_X * REG_DB)

// 如果是寄存器求外积，那么两个宏分别表示 单线程处理的 A列的元素个数和B行的元素个数
#define REG_PART_Y GEMM_BLOCK_Y    
#define REG_PART_X GEMM_BLOCK_Y 

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
            // A[i][j] = 1;   
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
            // B[i][j] = 1; 
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

__inline__ __device__ double warpReduce(double localSum){
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);
    return localSum;
}


// 改 A行 B列 的内积
// 为 A列 B行 的外积
// 那么每个线程就从 v15的处理A一行的连续几个元素，变成v16的 处理A的一列和B的第tiy行
__global__ void DGEMM_v16(double * A, double * B, double * C){
    // smem的block上限是128*128*3=3*128*128字节 一个double8个字节 所以一个block的smem最多存 3*16*128个double
    // double buffering + AB数组对半分的情况下 一次处理的A矩阵元素个数为3*4*128个double
    // 用满的情况下，block/SM limit SMEM就是2
    // 用到2*4*128个double，block/SM limit SMEM就是3 受限于SM SMEM总量102400
    __shared__ double A_part[2][GEMM_BLOCK_Y][SMEM_PART_X], B_part[2][GEMM_BLOCK_Y][SMEM_PART_X];

    // 如果启动 2个block；一个block 占 256*128个寄存器 存256*64个double
    // 如果启动 4个block；一个block 占 256*64个寄存器 存256*32个double
    // 2*(REG_PART_Y+REG_PART_X) + REG_PART_Y*REG_PART_X
    volatile double A_part_reg[2][REG_PART_Y];
    volatile double B_part_reg[2][REG_PART_X];
    volatile double C_part_reg[REG_PART_Y][REG_PART_X];

    for(int i = 0; i < REG_PART_Y; i++){
        A_part_reg[0][i] = 0.0;
        A_part_reg[1][i] = 0.0;
        
    }
    for(int i = 0; i < REG_PART_X; i++){
        B_part_reg[0][i] = 0.0;
        B_part_reg[1][i] = 0.0;
    }
    for(int i = 0; i < REG_PART_Y; i++){
        for(int j = 0; j < REG_PART_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }
    // 线程组索引
    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;
    // K至少得是 SMEM_PART_X 所以 这里应该有一个if判断 让最后一轮多余的warps continue
    int num_loops = K / SMEM_PART_X;

     //// 读第0个SMEM块
     int A_glo_x = tix;
     int A_glo_y = biy * GEMM_BLOCK_Y + tiy;
     int B_glo_y = bix * GEMM_BLOCK_Y + tiy;
     for(int i = 0; i < REG_DB; i++){
         A_part[0][tiy][tix+i*GEMM_BLOCK_X] = A[A_glo_y * K + A_glo_x + i*GEMM_BLOCK_X];
         B_part[0][tiy][tix+i*GEMM_BLOCK_X] = B[B_glo_y * K + A_glo_x + i*GEMM_BLOCK_X];
     }
     __syncthreads();

    for(int i = 1; i < num_loops; i++){
        //// 读第i%2个SMEM块
        A_glo_x = i * SMEM_PART_X + tix;
        for(int reg_db = 0; reg_db < REG_DB; reg_db++){
            A_part[i%2][tiy][tix+reg_db*GEMM_BLOCK_X] = A[A_glo_y * K + A_glo_x + reg_db*GEMM_BLOCK_X];
            B_part[i%2][tiy][tix+reg_db*GEMM_BLOCK_X] = B[B_glo_y * K + A_glo_x + reg_db*GEMM_BLOCK_X];
        }
        
        //// 算第(i-1)%2个SMEM块

        // 只用tiy==0的那个线程束
        if(tiy == 0){

            // 读第0个A_regs
            for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                A_part_reg[0][reg_y] = A_part[(i-1)%2][reg_y][tix];
            }
            // 读第0个B_regs
            for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                B_part_reg[0][reg_x] = B_part[(i-1)%2][reg_x][tix];
            }

            // 读第j%2个A_regs和B_regs 算第j-1个C_regs
            for(int j = 1; j < REG_DB; j++){
                for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                    A_part_reg[j%2][reg_y] = A_part[(i-1)%2][reg_y][tix+j*GEMM_BLOCK_X];
                }
                for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                    B_part_reg[j%2][reg_x] = B_part[(i-1)%2][reg_x][tix+j*GEMM_BLOCK_X];
                }
                for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                    for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                        C_part_reg[reg_y][reg_x] += A_part_reg[(j-1)%2][reg_y] * B_part_reg[(j-1)%2][reg_x];
                    }
                }
            }
            // 算第(REG_DB-1)个C_regs
            for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                    C_part_reg[reg_y][reg_x] += A_part_reg[(REG_DB-1)%2][reg_y] * B_part_reg[(REG_DB-1)%2][reg_x];
                }
            }
        }
        __syncthreads();
    }

    //// 算第(num_loops-1)%2个SMEM块

    if(tiy == 0){

        // 读第0个A_regs
        for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
            A_part_reg[0][reg_y] = A_part[(num_loops-1)%2][reg_y][tix];
        }
        // 读第0个B_regs
        for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
            B_part_reg[0][reg_x] = B_part[(num_loops-1)%2][reg_x][tix];
        }

        // 读第j%2个A_regs和B_regs 算第j-1个C_regs
        for(int j = 1; j < REG_DB; j++){
            for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                A_part_reg[j%2][reg_y] = A_part[(num_loops-1)%2][reg_y][tix+j*GEMM_BLOCK_X];
            }
            for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                B_part_reg[j%2][reg_x] = B_part[(num_loops-1)%2][reg_x][tix+j*GEMM_BLOCK_X];
            }
            for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
                for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                    C_part_reg[reg_y][reg_x] += A_part_reg[(j-1)%2][reg_y] * B_part_reg[(j-1)%2][reg_x];
                }
            }
        }
        // 算第(REG_DB-1)个C_regs
        for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
            for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                C_part_reg[reg_y][reg_x] += A_part_reg[(REG_DB-1)%2][reg_y] * B_part_reg[(REG_DB-1)%2][reg_x];
            }
        }

        for(int reg_y = 0; reg_y < REG_PART_Y; reg_y++){
            for(int reg_x = 0; reg_x < REG_PART_X; reg_x++){
                C_part_reg[reg_y][reg_x] = warpReduce(C_part_reg[reg_y][reg_x]);
                int C_glo_y = bdy * biy + reg_y;
                int C_glo_x = bdy * bix + reg_x;
                C[C_glo_y * N + C_glo_x] = C_part_reg[reg_y][reg_x];
            }
        }

    }
}


__host__ void callDGEMM(double * d_A, double * d_B_T, double * d_C, double * d_B){
    int kernelIdx = 16;
    size_t C_Bytes = sizeof(double) * M * N;
    dim3 blockSize(1);
    dim3 gridSize(1);
    switch(kernelIdx){
        case 16:
            blockSize = {GEMM_BLOCK_X, GEMM_BLOCK_Y};
            gridSize = {N/GEMM_BLOCK_Y, M/GEMM_BLOCK_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            DGEMM_v16<<<gridSize, blockSize>>>(d_A, d_B_T, d_C);
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
                std::cout << "err: [" << i << ", "  << j << "] " << h_C[i*N+j]  << " - " << h_C_cublas[i*N+j] << " = " << err << std::endl;
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