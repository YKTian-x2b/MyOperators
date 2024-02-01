#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <random>
#include <ctime>
#include "cublas_v2.h"

#define M 1
#define K 1024
#define N 1024

#define BLK_SIZE 256

#define SMEM_Y 128
#define SMEM_X 8
#define REG_Y 8
#define REG_X REG_Y

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

float A[M][K];
float B[K][N];
float C[N];
const float alpha = 1.0;
const float beta = 0.0;

__host__ void assignData(){
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(1.5, 4.5);
    // std::cout << "Matrix A: " << std::endl;
    for(size_t i = 0; i < M; i++){
        for(size_t j = 0; j < K; j++){          
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
    for(size_t i = 0; i < K; i++){
        for(size_t j = 0; j < N; j++){
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


__global__ void sgemm(float *A, float *B, float *C){
    __shared__ float B_part[2][SMEM_X][SMEM_Y];
    float A_part_reg[REG_Y];
    float B_part_reg[2][REG_X];
    float C_part_reg[REG_X];
 
    int tix = threadIdx.x;
    int bix = blockIdx.x;
    int biy = blockIdx.y;
    int warp_id = tix >> 5;
    int lane_id = tix & 31;
    // const size_t tile_index_a = (warp_id/4)*32 + ((lane_id%16)/2)*4;
    size_t A_lds_x = biy * SMEM_Y + ((warp_id >> 2) << 5)  | ((lane_id & 14) << 1);
    // const size_t tile_index_b = (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    size_t B_lds_x = ((warp_id & 3) << 4) | ((lane_id >> 4) << 3) | ((lane_id & 1) << 2);
    // 外层循环K
    size_t num_loops_shared = K/SMEM_X;
    // 内存循环k 循环展开
    size_t num_loops_regs = SMEM_X;

    // glo -> reg A
    if(warp_id == 0){
        FETCH_FLOAT4(A_part_reg[0]) = FETCH_FLOAT4(A[A_lds_x]);
        FETCH_FLOAT4(A_part_reg[4]) = FETCH_FLOAT4(A[A_lds_x+64]);
    }
    // glo -> smem B
    size_t B_ldg_y_stride = BLK_SIZE * 4 / SMEM_Y;
    #pragma unroll
    for(size_t ldg_cnt = 0; ldg_cnt < SMEM_X; ldg_cnt += B_ldg_y_stride){
        size_t B_ldg_y = ldg_cnt + (tix << 2) / SMEM_Y;
        size_t B_ldg_x = bix * SMEM_Y + (tix << 2) % SMEM_Y;
        size_t B_sts_y = ldg_cnt + (tix << 2) / SMEM_Y;
        size_t B_sts_x = (tix << 2) % SMEM_Y;
        FETCH_FLOAT4(B_part[0][B_sts_y][B_sts_x]) = FETCH_FLOAT4(B[B_ldg_y * N + B_ldg_x]);
    }
    __syncthreads();

    for(size_t i = 1; i < num_loops_shared; i++){
        /// 第 i%2 个SMEM读
        #pragma unroll
        for(size_t ldg_cnt = 0; ldg_cnt < SMEM_X; ldg_cnt += B_ldg_y_stride){
            size_t B_ldg_y = i * SMEM_X + ldg_cnt + (tix << 2) / SMEM_Y;
            size_t B_ldg_x = bix * SMEM_Y + (tix << 2) % SMEM_Y;
            size_t B_sts_y = ldg_cnt + (tix << 2) / SMEM_Y;
            size_t B_sts_x = (tix << 2) % SMEM_Y;
            FETCH_FLOAT4(B_part[i%2][B_sts_y][B_sts_x]) = FETCH_FLOAT4(B[B_ldg_y * N + B_ldg_x]);
        }

        /// 第 (i-1)%2 个SMEM算
        // 第 0 个 regs读
        FETCH_FLOAT4(B_part_reg[0][0]) = FETCH_FLOAT4(B_part[(i-1)%2][0][B_lds_x]);
        FETCH_FLOAT4(B_part_reg[0][4]) = FETCH_FLOAT4(B_part[(i-1)%2][0][B_lds_x+64]);

        #pragma unroll
        for(size_t k = 1; k < num_loops_regs; k++){
            // 第 k%2 个regs读
            FETCH_FLOAT4(B_part_reg[k%2][0]) = FETCH_FLOAT4(B_part[(i-1)%2][k][B_lds_x]);
            FETCH_FLOAT4(B_part_reg[k%2][4]) = FETCH_FLOAT4(B_part[(i-1)%2][k][B_lds_x+64]);

            // 第 (k-1)%2 个regs算
            #pragma unroll
            for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
                #pragma unroll
                for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
                    // 待优化
                    C_part_reg[0+reg_cnt_a][0+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                    C_part_reg[0+reg_cnt_a][4+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(k-1)%2][4+reg_cnt_b];
                    C_part_reg[4+reg_cnt_a][0+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                    C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(k-1)%2][4+reg_cnt_b];
                }
            }
        } 
        // 第 (num_loops_regs-1)%2 个regs算
        #pragma unroll
        for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
            #pragma unroll
            for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
                C_part_reg[0+reg_cnt_a][0+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
                C_part_reg[0+reg_cnt_a][4+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
                C_part_reg[4+reg_cnt_a][0+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
                C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
            }
        }
        __syncthreads();
    }

    /// 第 (num_loops_shared-1)%2 个SMEM算
    // 第 0 个 regs读
    FETCH_FLOAT4(B_part_reg[0][0]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][0][B_lds_x]);
    FETCH_FLOAT4(B_part_reg[0][4]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][0][B_lds_x+64]);

    #pragma unroll
    for(size_t k = 1; k < num_loops_regs; k++){
        // 第 k%2 个regs读
        FETCH_FLOAT4(B_part_reg[k%2][0]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][k][B_lds_x]);
        FETCH_FLOAT4(B_part_reg[k%2][4]) = FETCH_FLOAT4(B_part[(num_loops_shared-1)%2][k][B_lds_x+64]);

        // 第 (k-1)%2 个regs算
        #pragma unroll
        for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
            #pragma unroll
            for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
                C_part_reg[reg_cnt_a][reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                C_part_reg[reg_cnt_a][4 + reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(k-1)%2][4 + reg_cnt_b];
                C_part_reg[4 + reg_cnt_a][reg_cnt_b] += A_part_reg[4 + reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(k-1)%2][4+reg_cnt_b];
            }
        }
    } 
    // 第 (num_loops_regs-1)%2 个regs算
    #pragma unroll
    for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
        #pragma unroll
        for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
            C_part_reg[0+reg_cnt_a][0+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
            C_part_reg[0+reg_cnt_a][4+reg_cnt_b] += A_part_reg[reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
            C_part_reg[4+reg_cnt_a][0+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
            C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
        }
    }

    //// 写回
    #pragma unroll
    for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
        size_t C_glo_y = biy * SMEM_Y + A_lds_x + reg_cnt_a;
        size_t C_glo_x = bix * SMEM_Y + B_lds_x;
        FETCH_FLOAT4(C[C_glo_y * N + C_glo_x]) = FETCH_FLOAT4(C_part_reg[reg_cnt_a][0]);
        FETCH_FLOAT4(C[C_glo_y * N + C_glo_x + 64]) = FETCH_FLOAT4(C_part_reg[reg_cnt_a][4]);
        FETCH_FLOAT4(C[(C_glo_y + 64) * N + C_glo_x]) = FETCH_FLOAT4(C_part_reg[4+reg_cnt_a][0]);
        FETCH_FLOAT4(C[(C_glo_y + 64) * N + C_glo_x + 64]) = FETCH_FLOAT4(C_part_reg[4+reg_cnt_a][4]);
    }

}


__global__ void sgemm2(float *A, float *B, float *C){
    
}


__host__ void callSGEMM(float * d_A, float * d_B, float * d_C){
    size_t C_Bytes = sizeof(float) * M * N;

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
    // for(size_t i = 0; i < M; i++){
    //     for(size_t j = 0; j < N; j++){
    //         std::cout << h_C_cublas[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;


    dim3 blockSize = {BLK_SIZE};
    dim3 gridSize = {N/SMEM_Y, M/SMEM_Y};
    printf("gridSize:  (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
    printf("blockSize: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
    sgemm<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    printf("SGEMM after: %s\n", cudaGetErrorString(cudaGetLastError()));
    float * h_C = (float *)malloc(C_Bytes);
    CHECK(cudaMemcpy(h_C, d_C, C_Bytes, cudaMemcpyDeviceToHost));

    // std::cout << "Matrix C: " << std::endl;
    // for(size_t i = 0; i < M; i++){
    //     for(size_t j = 0; j < N; j++){
    //         std::cout << h_C[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

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

// nvcc -DVALID -I/opt/kaiProjects/GEMM_kai/Utils -L /usr/local/cuda/lib64 -l cuda -l cublas -o res/sgemm sgemm.cu