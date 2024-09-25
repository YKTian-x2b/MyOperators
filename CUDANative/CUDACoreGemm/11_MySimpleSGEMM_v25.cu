#include "../Common.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// #define VALID

#ifdef VALID

#define M 256
#define N 256
#define K 2048

#else

#define M 2048
#define N 2048
#define K 2048

#endif

#define BLOCK_Y 16
#define BLOCK_X 16

#define SMEM_Y 128
#define SMEM_X 16
#define REG_Y 8
#define REG_X REG_Y

#define NUM_SMEM_DATA_PER_THREAD ((SMEM_Y * SMEM_X) / (BLOCK_Y * BLOCK_X))

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
            // A[i][j] = 1; 
            A[i][j] = (i*2+j)*0.001; 
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
            // B[i][j] = 1; 
            B[i][j] = (i*2+j)*0.001;
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

// 写一个有容错的 能改变 SMEM_和REG_
// 这个时候能有差不多 cublas90%的性能了
// 但是容错还不够，SMEM_Y/REG_Y 必须是16 还能再写写 把reg的8 改成 4
__global__ void SGEMM_v25(float *A, float *B, float *C){
    __shared__ float A_part[2][SMEM_X][SMEM_Y], B_part[2][SMEM_X][SMEM_Y];

    float A_trans_reg[NUM_SMEM_DATA_PER_THREAD];

    float A_part_reg[2][REG_Y];
    float B_part_reg[2][REG_X];
    float C_part_reg[REG_Y][REG_X];
    #pragma unroll
    for(int i = 0; i < REG_Y; i++){
        #pragma unroll
        for(int j = 0; j < REG_X; j++){
            C_part_reg[i][j] = 0.0;
        }
    }

    int tix = threadIdx.x, tiy = threadIdx.y;
    int bix = blockIdx.x, biy = blockIdx.y;
    int bdx = blockDim.x, bdy = blockDim.y;

    int num_loops_shared = K/SMEM_X;
    int num_loops_regs = SMEM_X;

    /// 第 0 个SMEM读
    int A_ldg_y = biy * SMEM_Y + (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD / SMEM_X;
    int A_ldg_x = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD % SMEM_X;
    int A_sts_trans_x = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD / SMEM_X;
    int A_sts_trans_y = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD % SMEM_X;
    #pragma unroll
    for(int reg_cnt = 0; reg_cnt < NUM_SMEM_DATA_PER_THREAD; reg_cnt += 4){
        FETCH_FLOAT4(A_trans_reg[reg_cnt]) = FETCH_FLOAT4(A[A_ldg_y * K + A_ldg_x + reg_cnt]);
        A_part[0][A_sts_trans_y+reg_cnt][A_sts_trans_x] = A_trans_reg[reg_cnt];
        A_part[0][A_sts_trans_y+reg_cnt+1][A_sts_trans_x] = A_trans_reg[reg_cnt + 1];
        A_part[0][A_sts_trans_y+reg_cnt+2][A_sts_trans_x] = A_trans_reg[reg_cnt + 2];
        A_part[0][A_sts_trans_y+reg_cnt+3][A_sts_trans_x] = A_trans_reg[reg_cnt + 3];
    }
    int B_ldg_y = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD / SMEM_Y;
    int B_ldg_x = bix * SMEM_Y + (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD % SMEM_Y;
    int B_sts_y = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD / SMEM_Y;
    int B_sts_x = (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD % SMEM_Y;
    #pragma unroll
    for(int reg_cnt = 0; reg_cnt < NUM_SMEM_DATA_PER_THREAD; reg_cnt += 4){
        FETCH_FLOAT4(B_part[0][B_sts_y][B_sts_x + reg_cnt]) = FETCH_FLOAT4(B[B_ldg_y * N + B_ldg_x + reg_cnt]);
    }
    
    __syncthreads();
    
    for(int i = 1; i < num_loops_shared; i++){
        /// 第 i%2 个SMEM读
        A_ldg_x = i * SMEM_X + (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD % SMEM_X;
        #pragma unroll
        for(int reg_cnt = 0; reg_cnt < NUM_SMEM_DATA_PER_THREAD; reg_cnt += 4){
            FETCH_FLOAT4(A_trans_reg[reg_cnt]) = FETCH_FLOAT4(A[A_ldg_y * K + A_ldg_x + reg_cnt]);
            A_part[i%2][A_sts_trans_y+reg_cnt][A_sts_trans_x] = A_trans_reg[reg_cnt];
            A_part[i%2][A_sts_trans_y+reg_cnt+1][A_sts_trans_x] = A_trans_reg[reg_cnt + 1];
            A_part[i%2][A_sts_trans_y+reg_cnt+2][A_sts_trans_x] = A_trans_reg[reg_cnt + 2];
            A_part[i%2][A_sts_trans_y+reg_cnt+3][A_sts_trans_x] = A_trans_reg[reg_cnt + 3];
        }
        B_ldg_y = i * SMEM_X + (tiy * BLOCK_X + tix) * NUM_SMEM_DATA_PER_THREAD / SMEM_Y;
        #pragma unroll
        for(int reg_cnt = 0; reg_cnt < NUM_SMEM_DATA_PER_THREAD; reg_cnt += 4){
            FETCH_FLOAT4(B_part[i%2][B_sts_y][B_sts_x + reg_cnt]) = FETCH_FLOAT4(B[B_ldg_y * N + B_ldg_x + reg_cnt]);
        }

        /// 第 (i-1)%2 个SMEM算
        
        // 第 0 个 regs读
        int A_lds_x = tiy * REG_Y;
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a += 4){
            FETCH_FLOAT4(A_part_reg[0][reg_cnt_a]) = FETCH_FLOAT4(A_part[(i-1)%2][0][A_lds_x + reg_cnt_a]);
        }
        int B_lds_x = tix * REG_X;
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b += 4){
            FETCH_FLOAT4(B_part_reg[0][reg_cnt_b]) = FETCH_FLOAT4(B_part[(i-1)%2][0][B_lds_x + reg_cnt_b]);
        }

        #pragma unroll
        for(int k = 1; k < num_loops_regs; k++){
            // 第 k%2 个regs读
            #pragma unroll
            for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a += 4){
                FETCH_FLOAT4(A_part_reg[k%2][reg_cnt_a]) = FETCH_FLOAT4(A_part[(i-1)%2][k][A_lds_x + reg_cnt_a]);
            }
            #pragma unroll
            for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b += 4){
                FETCH_FLOAT4(B_part_reg[k%2][reg_cnt_b]) = FETCH_FLOAT4(B_part[(i-1)%2][k][B_lds_x + reg_cnt_b]);
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
    int A_lds_x = tiy * REG_Y;
    #pragma unroll
    for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a += 4){
        FETCH_FLOAT4(A_part_reg[0][reg_cnt_a]) = FETCH_FLOAT4(A_part[(num_loops_shared-1)%2][0][A_lds_x + reg_cnt_a]);
    }
    int B_lds_x = tix * REG_X;
    #pragma unroll
    for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b += 4){
        FETCH_FLOAT4(B_part_reg[0][reg_cnt_b]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][0][B_lds_x + reg_cnt_b]);
    }

    #pragma unroll
    for(int k = 1; k < num_loops_regs; k++){
        // 第 k%2 个regs读
        #pragma unroll
        for(int reg_cnt_a = 0; reg_cnt_a < REG_Y; reg_cnt_a += 4){
            FETCH_FLOAT4(A_part_reg[k%2][reg_cnt_a]) = FETCH_FLOAT4(A_part[(num_loops_shared-1)%2][k][A_lds_x + reg_cnt_a]);
        }
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b += 4){
            FETCH_FLOAT4(B_part_reg[k%2][reg_cnt_b]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][k][B_lds_x + reg_cnt_b]);
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
        int C_glo_y = biy * SMEM_Y + tiy * REG_Y + reg_cnt_a;
        int C_glo_x = bix * SMEM_Y + tix * REG_X;
        #pragma unroll
        for(int reg_cnt_b = 0; reg_cnt_b < REG_X; reg_cnt_b+=4){
            FETCH_FLOAT4(C[C_glo_y*N+C_glo_x+reg_cnt_b]) = FETCH_FLOAT4(C_part_reg[reg_cnt_a][reg_cnt_b]);
        }
    }
}

__host__ void callSGEMM(float * d_A, float * d_B, float * d_C){
    int kernelIdx = 25;
    size_t C_Bytes = sizeof(float) * M * N;
    dim3 blockSize(1);
    dim3 gridSize(1);
    switch(kernelIdx){
        case 25:
            blockSize = {BLOCK_X, BLOCK_Y};
            gridSize = {N/SMEM_Y, M/SMEM_Y};
            printf("Kernel Num: %d\n", kernelIdx);
            printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
            printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
            SGEMM_v25<<<gridSize, blockSize>>>(d_A, d_B, d_C);
            break;
        default:
            break;
    }
    cudaDeviceSynchronize();
    printf("SGEMM after: %s\n", cudaGetErrorString(cudaGetLastError()));
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

// nvcc -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/11_res 11_MySimpleSGEMM_v25.cu

// nvcc --keep --keep-dir midRes -gencode=arch=compute_86,code=\"sm_86,compute_86\" -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/11_res 11_MySimpleSGEMM_v25.cu
// cuasm --bin2asm midRes/11_MySimpleSGEMM_v25.sm_86.cubin -o midRes/11_MySimpleSGEMM_v25.sm_86.cuasm