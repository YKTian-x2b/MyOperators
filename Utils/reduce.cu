#define BDIM 256

template<typename T, typename TResult>
__global__ void addBatchSumKernelV2(T *input, TResult *result, size_t stride, size_t n) {
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * stride + threadIdx.x;
    T *data_in = input + blockIdx.x * blockDim.x * stride;
    __shared__ TResult data_out[BDIM];

    // 这里的循环展开代码 应该随着 stride或UNROLL_NUM 的值而改变
    if (7 * blockDim.x + idx < n) {
        T a0 = data_in[tid];
        T a1 = data_in[1 * blockDim.x + tid];
        T a2 = data_in[2 * blockDim.x + tid];
        T a3 = data_in[3 * blockDim.x + tid];
        T a4 = data_in[4 * blockDim.x + tid];
        T a5 = data_in[5 * blockDim.x + tid];
        T a6 = data_in[6 * blockDim.x + tid];
        T a7 = data_in[7 * blockDim.x + tid];

        data_out[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) {
        data_out[tid] += data_out[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        data_out[tid] += data_out[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        data_out[tid] += data_out[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        data_out[tid] += data_out[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        volatile TResult *v_s_mem = data_out;
        v_s_mem[tid] += v_s_mem[tid + 32];
        v_s_mem[tid] += v_s_mem[tid + 16];
        v_s_mem[tid] += v_s_mem[tid + 8];
        v_s_mem[tid] += v_s_mem[tid + 4];
        v_s_mem[tid] += v_s_mem[tid + 2];
        v_s_mem[tid] += v_s_mem[tid + 1];
    }
    if (tid == 0) {
        result[bid] = data_out[0];
    }
}

__inline__ __device__ int warpReduce(int localSum)
{
    // 当前线程的lane_id异或16得到的线程号的线程将以返回值的形式得到第一个参数
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);
    return localSum;
}

// 这个没有循环展开
template<typename T, typename TResult>
__global__ void ReduceShfl(T *input, TResult *result, size_t n) {
    // shared mem for each warp sum
    __shared__ TResult smem[SMEMDIM];
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = bid * blockDim.x + tid;
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;

    if(idx >= n)    return;

    int mySum = input[idx];
    mySum = warpReduce(mySum);

    if(laneIdx == 0)    smem[warpIdx] = mySum;
    __syncthreads();
    // 一个block最多有1024个线程
    mySum = (tid < SMEMDIM) ? smem[laneIdx] : 0;
    if(warpIdx == 0)    mySum = warpReduce(mySum);

    if(tid == 0)    result[bid] = mySum;
}