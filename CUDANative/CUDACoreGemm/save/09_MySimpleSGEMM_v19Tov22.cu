#include "../Common.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"

// #define VALID

#ifdef VALID

#define M 256
#define N 256
#define K 2048

#else

#define M 512
#define N 512
#define K 2048

#endif

#define BLOCK_Y 16
#define BLOCK_X 16

#define SMEM_Y 128
#define SMEM_X 8
#define REG_Y 8
#define REG_X 8

float A[M][K], B[K][N], C[M][N];
const float alpha = 1.0;
const float beta = 0.0;

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(1.5, 4.5);
    // std::cout << "Matrix A: " << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){          
#ifdef VALID
            A[i][j] = 1; 
            // A[i][j] = (i*2+j)*0.001; 
            // A[i][j] = u(e);
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
            B[i][j] = 1; 
            // B[i][j] = (i*2+j)*0.001;
            // B[i][j] = u(e);
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

// 测试用16*16的block 做global->shared的 128*8 传输的效率(不同方法下)
__global__ void SGEMM_v19(float *A, float *B, float *C){
    __shared__ float A_part[SMEM_Y][SMEM_X], B_part[SMEM_X][SMEM_Y];

    float A_part_reg[REG_Y];
    float B_part_reg[REG_X];
    float C_part_reg[REG_Y][REG_X];
    for(int i = 0; i < REG_Y; i++){
        A_part_reg[i] = 0.0;
    }
    for(int i = 0; i < REG_X; i++){
        B_part_reg[i] = 0.0;
    }
    for(int i = 0; i < REG_Y; i++){
        for(int j = 0; j < REG_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }

    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    int num_loops_shared = K/SMEM_X;
    int num_loops_per_smem = (SMEM_Y * SMEM_X) / (BLOCK_Y * BLOCK_X);
    for(int i = 0; i < num_loops_shared; i++){
        for(int j = 0; j < num_loops_per_smem; j++){
            // tiy相同的16个线程负责连续两行SMEM
            int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_glo_x = i * SMEM_X + (tiy * BLOCK_X + tix) % SMEM_X;
            int A_smem_y = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_smem_x = (tiy * BLOCK_X + tix) % SMEM_X;
            A_part[A_smem_y][A_smem_x] = A[A_glo_y * K + A_glo_x];
        }
        for(int j = 0; j < num_loops_per_smem; j++){
            int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_glo_y = i * SMEM_X + j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            B_part[B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
        }
        __syncthreads();
        
        int num_loops_regs = REG_X;
        for(int k = 0; k < num_loops_regs; k++){
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                int A_smem_y = tiy * REG_Y + reg_cnt_a;
                int A_smem_x = k;
                A_part_reg[reg_cnt_a] = A_part[A_smem_y][A_smem_x];
            }
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                int B_smem_y = k;
                int B_smem_x = tix * REG_X + reg_cnt_b;
                B_part_reg[reg_cnt_b] = B_part[B_smem_y][B_smem_x];
            }
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                    C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[reg_cnt_b];
                }
            }
        } 
    }

    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int C_glo_y = biy * SMEM_Y + tiy * REG_Y + reg_cnt_a;
            int C_glo_x = bix * SMEM_Y + tix * REG_X + reg_cnt_b;
            C[C_glo_y*N+C_glo_x] = C_part_reg[reg_cnt_a][reg_cnt_b];
        }
    }
}

__global__ void SGEMM_v20(float *A, float *B, float *C){
    __shared__ float A_part[SMEM_X][SMEM_Y], B_part[SMEM_X][SMEM_Y];

    float A_part_reg[REG_Y];
    float B_part_reg[REG_X];
    float C_part_reg[REG_Y][REG_X];
    for(int i = 0; i < REG_Y; i++){
        A_part_reg[i] = 0.0;
    }
    for(int i = 0; i < REG_X; i++){
        B_part_reg[i] = 0.0;
    }
    for(int i = 0; i < REG_Y; i++){
        for(int j = 0; j < REG_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }

    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    int num_loops_shared = K/SMEM_X;
    int num_loops_per_smem = (SMEM_Y * SMEM_X) / (BLOCK_Y * BLOCK_X);
    for(int i = 0; i < num_loops_shared; i++){
        for(int j = 0; j < num_loops_per_smem; j++){
            // tiy相同的16个线程负责连续两行SMEM
            int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_glo_x = i * SMEM_X + (tiy * BLOCK_X + tix) % SMEM_X;
            int A_smem_trans_x = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_smem_trans_y = (tiy * BLOCK_X + tix) % SMEM_X;
            A_part[A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
        }
        for(int j = 0; j < num_loops_per_smem; j++){
            int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_glo_y = i * SMEM_X + j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            B_part[B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
        }
        __syncthreads();
        
        int num_loops_regs = REG_X;
        for(int k = 0; k < num_loops_regs; k++){
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                int A_smem_y = k;
                int A_smem_x = tiy * REG_Y + reg_cnt_a;
                A_part_reg[reg_cnt_a] = A_part[A_smem_y][A_smem_x];
            }
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                int B_smem_y = k;
                int B_smem_x = tix * REG_X + reg_cnt_b;
                B_part_reg[reg_cnt_b] = B_part[B_smem_y][B_smem_x];
            }
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                    C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[reg_cnt_b];
                }
            }
        } 
    }

    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int C_glo_y = biy * SMEM_Y + tiy * REG_Y + reg_cnt_a;
            int C_glo_x = bix * SMEM_Y + tix * REG_X + reg_cnt_b;
            C[C_glo_y*N+C_glo_x] = C_part_reg[reg_cnt_a][reg_cnt_b];
        }
    }
}

__global__ void SGEMM_v21(float *A, float *B, float *C){
    __shared__ float A_part[2][SMEM_X][SMEM_Y], B_part[2][SMEM_X][SMEM_Y];

    float A_part_reg[2][REG_Y];
    float B_part_reg[2][REG_X];
    float C_part_reg[REG_Y][REG_X];
    #pragma unroll
    for(int i = 0; i < REG_Y; i++){
        A_part_reg[0][i] = 0.0;
        A_part_reg[1][i] = 0.0;
    }
    #pragma unroll
    for(int i = 0; i < REG_X; i++){
        B_part_reg[0][i] = 0.0;
        B_part_reg[1][i] = 0.0;
    }
    #pragma unroll
    for(int i = 0; i < REG_Y; i++){
        for(int j = 0; j < REG_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }

    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    int num_loops_shared = K/SMEM_X;
    int num_loops_per_smem = (SMEM_Y * SMEM_X) / (BLOCK_Y * BLOCK_X);
    int num_loops_regs = REG_X;

    /// 第 0 个SMEM读
    #pragma unroll
    for(int j = 0; j < num_loops_per_smem; j++){
        // tiy相同的16个线程负责连续两行SMEM
        int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
        int A_glo_x = (tiy * BLOCK_X + tix) % SMEM_X;
        int A_smem_trans_x = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
        int A_smem_trans_y = (tiy * BLOCK_X + tix) % SMEM_X;
        A_part[0][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
    }
    #pragma unroll
    for(int j = 0; j < num_loops_per_smem; j++){
        int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
        int B_glo_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
        int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
        int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
        B_part[0][B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
    }
    __syncthreads();

    #pragma unroll
    for(int i = 1; i < num_loops_shared; i++){
        /// 第 i%2 个SMEM读
        #pragma unroll
        for(int j = 0; j < num_loops_per_smem; j++){
            // tiy相同的16个线程负责连续两行SMEM
            int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_glo_x = i * SMEM_X + (tiy * BLOCK_X + tix) % SMEM_X;
            int A_smem_trans_x = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
            int A_smem_trans_y = (tiy * BLOCK_X + tix) % SMEM_X;
            A_part[i%2][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
        }
        #pragma unroll
        for(int j = 0; j < num_loops_per_smem; j++){
            int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_glo_y = i * SMEM_X + j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            B_part[i%2][B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
        }
        
        /// 第 (i-1)%2 个SMEM算
        
        // 第 0 个 regs读
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            int A_smem_y = 0;
            int A_smem_x = tiy * REG_Y + reg_cnt_a;
            A_part_reg[0][reg_cnt_a] = A_part[(i-1)%2][A_smem_y][A_smem_x];
        }
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int B_smem_y = 0;
            int B_smem_x = tix * REG_X + reg_cnt_b;
            B_part_reg[0][reg_cnt_b] = B_part[(i-1)%2][B_smem_y][B_smem_x];
        }

        #pragma unroll
        for(int k = 1; k < num_loops_regs; k++){
            // 第 k%2 个regs读
            #pragma unroll
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                int A_smem_y = k;
                int A_smem_x = tiy * REG_Y + reg_cnt_a;
                A_part_reg[k%2][reg_cnt_a] = A_part[(i-1)%2][A_smem_y][A_smem_x];
            }
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                int B_smem_y = k;
                int B_smem_x = tix * REG_X + reg_cnt_b;
                B_part_reg[k%2][reg_cnt_b] = B_part[(i-1)%2][B_smem_y][B_smem_x];
            }

            // 第 (k-1)%2 个regs算
            #pragma unroll
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                    C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                }
            }
        } 

        // 第 (num_loops_regs-1)%2 个regs算
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
            }
        }

        __syncthreads();
    }

    /// 第 (num_loops_shared-1)%2 个SMEM算
    // 第 0 个 regs读
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        int A_smem_y = 0;
        int A_smem_x = tiy * REG_Y + reg_cnt_a;
        A_part_reg[0][reg_cnt_a] = A_part[(num_loops_shared-1)%2][A_smem_y][A_smem_x];
    }
    #pragma unroll
    for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
        int B_smem_y = 0;
        int B_smem_x = tix * REG_X + reg_cnt_b;
        B_part_reg[0][reg_cnt_b] = B_part[(num_loops_shared-1)%2][B_smem_y][B_smem_x];
    }
    #pragma unroll
    for(int k = 1; k < num_loops_regs; k++){
        // 第 k%2 个regs读
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            int A_smem_y = k;
            int A_smem_x = tiy * REG_Y + reg_cnt_a;
            A_part_reg[k%2][reg_cnt_a] = A_part[(num_loops_shared-1)%2][A_smem_y][A_smem_x];
        }
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int B_smem_y = k;
            int B_smem_x = tix * REG_X + reg_cnt_b;
            B_part_reg[k%2][reg_cnt_b] = B_part[(num_loops_shared-1)%2][B_smem_y][B_smem_x];
        }

        // 第 (k-1)%2 个regs算
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
            }
        }
    } 

    // 第 (num_loops_regs-1)%2 个regs算
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
        }
    }

    //// 写回
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int C_glo_y = biy * SMEM_Y + tiy * REG_Y + reg_cnt_a;
            int C_glo_x = bix * SMEM_Y + tix * REG_X + reg_cnt_b;
            C[C_glo_y*N+C_glo_x] = C_part_reg[reg_cnt_a][reg_cnt_b];
        }
    }
}

// 改变矩阵A glo->shared的传输方式
__global__ void SGEMM_v22(float *A, float *B, float *C){
    __shared__ float A_part[2][SMEM_X][SMEM_Y], B_part[2][SMEM_X][SMEM_Y];
    // SM共65536个寄存器 启动2个block的话 256的block 一个线程只能用128个寄存器 目前这个kernel就是占了128个寄存器
    // 但是只能启动1个block 可能是还有一些的使用没算进来，导致的。所以，尝试降低寄存器的使用量。
    float A_part_reg[2][REG_Y];
    float B_part_reg[2][REG_X];
    float C_part_reg[REG_Y][REG_X];
    #pragma unroll
    for(int i = 0; i < REG_Y; i++){
        A_part_reg[0][i] = 0.0;
        A_part_reg[1][i] = 0.0;
    }
    #pragma unroll
    for(int i = 0; i < REG_X; i++){
        B_part_reg[0][i] = 0.0;
        B_part_reg[1][i] = 0.0;
    }
    #pragma unroll
    for(int i = 0; i < REG_Y; i++){
        for(int j = 0; j < REG_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }

    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    int num_loops_shared = K/SMEM_X;
    int num_loops_per_smem = (SMEM_Y * SMEM_X) / (BLOCK_Y * BLOCK_X);
    int num_shared_data_per_thread = num_loops_per_smem;
    int num_loops_regs = REG_X;

    /// 第 0 个SMEM读

    // #pragma unroll
    // for(int j = 0; j < num_loops_per_smem; j++){
    //     // tiy相同的16个线程负责连续两行SMEM
    //     int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
    //     int A_glo_x = (tiy * BLOCK_X + tix) % SMEM_X;
    //     int A_smem_trans_x = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
    //     int A_smem_trans_y = (tiy * BLOCK_X + tix) % SMEM_X;
    //     A_part[0][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
    // }
    
    // 新的A glo->shared的传输方式 
    for(int j = 0; j < num_shared_data_per_thread; j++){
        int A_glo_y = biy * SMEM_Y + (tiy * BLOCK_X + tix) * num_shared_data_per_thread / SMEM_X;
        int A_glo_x = (tix % 2) * num_shared_data_per_thread + j;
        int A_smem_trans_x = (tiy * BLOCK_X + tix) * num_shared_data_per_thread / SMEM_X;
        int A_smem_trans_y = (tix % 2) * num_shared_data_per_thread + j;
        A_part[0][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
    }

    // #pragma unroll
    for(int j = 0; j < num_loops_per_smem; j++){
        int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
        int B_glo_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
        int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
        int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
        B_part[0][B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
    }
    __syncthreads();

    
    for(int i = 1; i < num_loops_shared; i++){
        /// 第 i%2 个SMEM读

        // #pragma unroll
        // for(int j = 0; j < num_loops_per_smem; j++){
        //     // tiy相同的16个线程负责连续两行SMEM
        //     int A_glo_y = biy * SMEM_Y + j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
        //     int A_glo_x = i * SMEM_X + (tiy * BLOCK_X + tix) % SMEM_X;
        //     int A_smem_trans_x = j * (SMEM_Y/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_X;
        //     int A_smem_trans_y = (tiy * BLOCK_X + tix) % SMEM_X;
        //     A_part[i%2][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
        // }
        for(int j = 0; j < num_shared_data_per_thread; j++){
            int A_glo_y = biy * SMEM_Y + (tiy * BLOCK_X + tix) * num_shared_data_per_thread / SMEM_X;
            int A_glo_x = i * SMEM_X + (tix % 2) * num_shared_data_per_thread + j;
            int A_smem_trans_x = (tiy * BLOCK_X + tix) * num_shared_data_per_thread / SMEM_X;
            int A_smem_trans_y = (tix % 2) * num_shared_data_per_thread + j;
            A_part[i%2][A_smem_trans_y][A_smem_trans_x] = A[A_glo_y * K + A_glo_x];
        }

        // #pragma unroll
        for(int j = 0; j < num_loops_per_smem; j++){
            int B_glo_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_glo_y = i * SMEM_X + j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            int B_smem_x = (tiy * BLOCK_X + tix) % SMEM_Y;
            int B_smem_y = j * (SMEM_X/num_loops_per_smem) + (tiy * BLOCK_X + tix) / SMEM_Y;
            B_part[i%2][B_smem_y][B_smem_x] = B[B_glo_y * N + B_glo_x];
        }
        
        /// 第 (i-1)%2 个SMEM算
        
        // 第 0 个 regs读
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            int A_smem_x = tiy * REG_Y + reg_cnt_a;
            A_part_reg[0][reg_cnt_a] = A_part[(i-1)%2][0][A_smem_x];
        }
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int B_smem_x = tix * REG_X + reg_cnt_b;
            B_part_reg[0][reg_cnt_b] = B_part[(i-1)%2][0][B_smem_x];
        }

        #pragma unroll
        for(int k = 1; k < num_loops_regs; k++){
            // 第 k%2 个regs读
            #pragma unroll
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                int A_smem_x = tiy * REG_Y + reg_cnt_a;
                A_part_reg[k%2][reg_cnt_a] = A_part[(i-1)%2][k][A_smem_x];
            }
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                int B_smem_x = tix * REG_X + reg_cnt_b;
                B_part_reg[k%2][reg_cnt_b] = B_part[(i-1)%2][k][B_smem_x];
            }

            // 第 (k-1)%2 个regs算
            #pragma unroll
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
                #pragma unroll
                for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                    C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                }
            }
        } 
        // 第 (num_loops_regs-1)%2 个regs算
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
            }
        }
        __syncthreads();
    }

    /// 第 (num_loops_shared-1)%2 个SMEM算
    // 第 0 个 regs读
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        int A_smem_x = tiy * REG_Y + reg_cnt_a;
        A_part_reg[0][reg_cnt_a] = A_part[(num_loops_shared-1)%2][0][A_smem_x];
    }
    #pragma unroll
    for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
        int B_smem_x = tix * REG_X + reg_cnt_b;
        B_part_reg[0][reg_cnt_b] = B_part[(num_loops_shared-1)%2][0][B_smem_x];
    }
    #pragma unroll
    for(int k = 1; k < num_loops_regs; k++){
        // 第 k%2 个regs读
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            int A_smem_x = tiy * REG_Y + reg_cnt_a;
            A_part_reg[k%2][reg_cnt_a] = A_part[(num_loops_shared-1)%2][k][A_smem_x];
        }
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int B_smem_x = tix * REG_X + reg_cnt_b;
            B_part_reg[k%2][reg_cnt_b] = B_part[(num_loops_shared-1)%2][k][B_smem_x];
        }

        // 第 (k-1)%2 个regs算
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
                C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
            }
        }
    } 
    // 第 (num_loops_regs-1)%2 个regs算
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
        }
    }

    //// 写回
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a++){
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b++){
            int C_glo_y = biy * SMEM_Y + tiy * REG_Y + reg_cnt_a;
            int C_glo_x = bix * SMEM_Y + tix * REG_X + reg_cnt_b;
            C[C_glo_y*N+C_glo_x] = C_part_reg[reg_cnt_a][reg_cnt_b];
        }
    }
}

__host__ void callDGEMM(float * d_A, float * d_B, float * d_C){
    int kernelIdx = 22;
    size_t C_Bytes = sizeof(float) * M * N;
    dim3 blockSize(1);
    dim3 gridSize(1);
    switch(kernelIdx){
        case 19:
            blockSize = {BLOCK_X, BLOCK_Y};
            gridSize = {N/SMEM_Y, M/SMEM_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            SGEMM_v19<<<gridSize, blockSize>>>(d_A, d_B, d_C);
            break;
        case 20:
            blockSize = {BLOCK_X, BLOCK_Y};
            gridSize = {N/SMEM_Y, M/SMEM_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            SGEMM_v20<<<gridSize, blockSize>>>(d_A, d_B, d_C);
            break;
        case 21:
            blockSize = {BLOCK_X, BLOCK_Y};
            gridSize = {N/SMEM_Y, M/SMEM_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            SGEMM_v21<<<gridSize, blockSize>>>(d_A, d_B, d_C);
            break;
        case 22:
            blockSize = {BLOCK_X, BLOCK_Y};
            gridSize = {N/SMEM_Y, M/SMEM_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            SGEMM_v22<<<gridSize, blockSize>>>(d_A, d_B, d_C);
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
    printf("DGEMM after: %s\n", cudaGetErrorString(cudaGetLastError()));
    float * h_C = (float *)malloc(C_Bytes);
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
    float * d_C_cublas, * h_C_cublas;
    CHECK(cudaMalloc(&d_C_cublas, C_Bytes));
    h_C_cublas = (float *)malloc(C_Bytes);
    // cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M);
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
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
            float err = h_C[i*N+j] - h_C_cublas[i*N+j];
            if(fabs(err) >=  0.1){
                err_cnt++;
                wrong = true;
                std::cout << "err: [" << i << ", "  << j << "] " << h_C[i*N+j]  << " - " << h_C_cublas[i*N+j] << " = " << err << std::endl;
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

    callDGEMM(d_A, d_B, d_C);

    CHECK(cudaFree(d_A)); 
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    return 0;
}