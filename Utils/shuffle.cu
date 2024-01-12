#include "../Common.cuh"

#define BDIMX 16
#define SEGM 4

__global__ void test_shfl_xor_array(int *d_out, int *d_in, int const mask){
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for(int i = 0; i < SEGM; i++)   value[i] = d_in[idx + i];

    value[0] = __shfl_xor(value[0], mask, BDIMX);
    value[1] = __shfl_xor(value[1], mask, BDIMX);
    value[2] = __shfl_xor(value[2], mask, BDIMX);
    value[3] = __shfl_xor(value[3], mask, BDIMX);

    __syncthreads();
    for(int i = 0; i < SEGM; i++)   d_out[idx + i] = value[i];
}



int main(){
    int size_bytes = sizeof(int) * BDIMX;
    int * h_in = (int*)malloc(size_bytes);
    int * h_out = (int*)malloc(size_bytes);
    for(int i = 0; i < BDIMX; i++){
        h_in[i] = i + 1;
    }
    int * d_in, *d_out;
    cudaMalloc(&d_in, size_bytes);
    cudaMalloc(&d_out, size_bytes);

    cudaMemcpy(d_in, h_in, size_bytes, cudaMemcpyHostToDevice);

    test_shfl_xor_array<<<1, BDIMX/SEGM>>>(d_out, d_in, 1);

    cudaMemcpy(h_out, d_out, size_bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < BDIMX; i++){
        printf("%d ", h_out[i]);
    }
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}