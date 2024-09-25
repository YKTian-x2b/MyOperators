#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define M 256
#define N 256
#define K 2048

#define BLOCK_XY 256
#define SMEM_Y 128
#define SMEM_X 16
#define REG_Y 8
#define REG_X REG_Y
// 8
#define NUM_SMEM_DATA_PER_THREAD ((SMEM_Y * SMEM_X) / BLOCK_XY)

float A[M][K], B[K][N], C[M][N];

__global__ void func(float *A, float *B, float *C){
    __shared__ float A_part[2][SMEM_X][SMEM_Y], B_part[2][SMEM_X][SMEM_Y];
    
    float A_trans_reg[NUM_SMEM_DATA_PER_THREAD];
    float A_part_reg[2][REG_Y];
    float B_part_reg[2][REG_X];
    float C_part_reg[REG_Y][REG_X];

    size_t tix = threadIdx.x;
    size_t bix = blockIdx.x, biy = blockIdx.y;
    size_t warp_id = tix / 32;
    size_t lane_id = tix % 32;
    // const size_t tile_index_a = (warp_id/4)*32 + ((lane_id%16)/2)*4;
    size_t A_lds_x = ((warp_id >> 2) << 5)  | ((lane_id & 14) << 1);
    // const size_t tile_index_b = (warp_id%4)*16 + (lane_id/16)*8 + (lane_id%2)*4;
    size_t B_lds_x = ((warp_id & 3) << 4) | ((lane_id >> 4) << 3) | ((lane_id & 1) << 2);

    size_t num_loops_shared = K/SMEM_X;
    size_t num_loops_regs = SMEM_X;

    size_t A_ldg_y_stride = BLOCK_XY * 4 / SMEM_X;
    #pragma unroll
    for(size_t ldg_cnt = 0; ldg_cnt < SMEM_Y; ldg_cnt += A_ldg_y_stride){
        size_t A_ldg_y = biy * SMEM_Y + ldg_cnt + (tix << 2) / SMEM_X;
        size_t A_ldg_x = (tix << 2) % SMEM_X;
        size_t A_sts_trans_x = ldg_cnt + (tix << 2) / SMEM_X;
        size_t A_sts_trans_y = (tix << 2) % SMEM_X;
        size_t A_trans_reg_idx = ldg_cnt / A_ldg_y_stride * 4;
        FETCH_FLOAT4(A_trans_reg[A_trans_reg_idx]) = FETCH_FLOAT4(A[A_ldg_y * K + A_ldg_x]);
        A_part[0][A_sts_trans_y][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx];
        A_part[0][A_sts_trans_y+1][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 1];
        A_part[0][A_sts_trans_y+2][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 2];
        A_part[0][A_sts_trans_y+3][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 3];
    }
    size_t B_ldg_y_stride = BLOCK_XY * 4 / SMEM_Y;     // 8
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
        for(size_t ldg_cnt = 0; ldg_cnt < SMEM_Y; ldg_cnt += A_ldg_y_stride){
            size_t A_ldg_y = biy * SMEM_Y + ldg_cnt + (tix << 2) / SMEM_X;
            size_t A_ldg_x = i * SMEM_X + (tix << 2) % SMEM_X;
            size_t A_sts_trans_x = ldg_cnt + (tix << 2) / SMEM_X;
            size_t A_sts_trans_y = (tix << 2) % SMEM_X;
            size_t A_trans_reg_idx = ldg_cnt / A_ldg_y_stride * 4;
            FETCH_FLOAT4(A_trans_reg[A_trans_reg_idx]) = FETCH_FLOAT4(A[A_ldg_y * K + A_ldg_x]);
            A_part[i%2][A_sts_trans_y][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx];
            A_part[i%2][A_sts_trans_y+1][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 1];
            A_part[i%2][A_sts_trans_y+2][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 2];
            A_part[i%2][A_sts_trans_y+3][A_sts_trans_x] = A_trans_reg[A_trans_reg_idx + 3];
        }
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
        FETCH_FLOAT4(A_part_reg[0][0]) = FETCH_FLOAT4(A_part[(i-1)%2][0][A_lds_x]);
        FETCH_FLOAT4(A_part_reg[0][4]) = FETCH_FLOAT4(A_part[(i-1)%2][0][A_lds_x+64]);
        FETCH_FLOAT4(B_part_reg[0][0]) = FETCH_FLOAT4(B_part[(i-1)%2][0][B_lds_x]);
        FETCH_FLOAT4(B_part_reg[0][4]) = FETCH_FLOAT4(B_part[(i-1)%2][0][B_lds_x+64]);
        
        #pragma unroll
        for(size_t k = 1; k < num_loops_regs; k++){
            // 第 k%2 个regs读
            FETCH_FLOAT4(A_part_reg[k%2][0]) = FETCH_FLOAT4(A_part[(i-1)%2][k][A_lds_x]);
            FETCH_FLOAT4(A_part_reg[k%2][4]) = FETCH_FLOAT4(A_part[(i-1)%2][k][A_lds_x+64]);
            FETCH_FLOAT4(B_part_reg[k%2][0]) = FETCH_FLOAT4(B_part[(i-1)%2][k][B_lds_x]);
            FETCH_FLOAT4(B_part_reg[k%2][4]) = FETCH_FLOAT4(B_part[(i-1)%2][k][B_lds_x+64]);

            // 第 (k-1)%2 个regs算
            #pragma unroll
            for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
                #pragma unroll
                for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
                    // 待优化
                    C_part_reg[0+reg_cnt_a][0+reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                    C_part_reg[0+reg_cnt_a][4+reg_cnt_b] += A_part_reg[(k-1)%2][reg_cnt_a] * B_part_reg[(k-1)%2][4+reg_cnt_b];
                    C_part_reg[4+reg_cnt_a][0+reg_cnt_b] += A_part_reg[(k-1)%2][4+reg_cnt_a] * B_part_reg[(k-1)%2][reg_cnt_b];
                    C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[(k-1)%2][4+reg_cnt_a] * B_part_reg[(k-1)%2][4+reg_cnt_b];
                }
            }
        } 
        // 第 (num_loops_regs-1)%2 个regs算
        #pragma unroll
        for(size_t reg_cnt_a = 0; reg_cnt_a < 4; reg_cnt_a++){
            #pragma unroll
            for(size_t reg_cnt_b = 0; reg_cnt_b < 4; reg_cnt_b++){
                C_part_reg[0+reg_cnt_a][0+reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
                C_part_reg[0+reg_cnt_a][4+reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
                C_part_reg[4+reg_cnt_a][0+reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][reg_cnt_b];
                C_part_reg[4+reg_cnt_a][4+reg_cnt_b] += A_part_reg[(num_loops_regs-1)%2][4+reg_cnt_a] * B_part_reg[(num_loops_regs-1)%2][4+reg_cnt_b];
            }
        }
        __syncthreads();
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

int main(){
    return 0;
}

// nvcc test_cuda3.cu -o res/test_cuda3 -gencode=arch=compute_86,code=\"sm_86,compute_86\"

// cuobjdump -ptx res/test_cuda3 > res/test_cuda3.ptx
// cuobjdump -sass res/test_cuda3 > res/test_cuda3.sass