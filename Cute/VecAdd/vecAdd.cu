#include <iostream>
#include "cuda_fp16.h"
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// z = ax+by+c
template<int ELTS_PER_THD=8>
__global__ void vec_add(half *z, int num, const half *x, const half *y, const half a, const half b, const half c){
    using namespace cute;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx * ELTS_PER_THD > num)    return;

    Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
    Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
    Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

    Tensor tzr = local_tile(tz, make_tile(Int<ELTS_PER_THD>{}), make_coord(idx));
    Tensor txr = local_tile(tx, make_tile(Int<ELTS_PER_THD>{}), make_coord(idx));
    Tensor tyr = local_tile(ty, make_tile(Int<ELTS_PER_THD>{}), make_coord(idx));

    Tensor txR = make_tensor_like(txr);
    Tensor tyR = make_tensor_like(tyr);
    Tensor tzR = make_tensor_like(tzr);

    copy(txr, txR);
    copy(tyr, tyR);

    // 为了用HFMA2
    half2 a2 = {a, a};
    half2 b2 = {b, b};
    half2 c2 = {c, c};
    auto txR2 = recast<half2>(txR);
    auto tyR2 = recast<half2>(tyR);
    auto tzR2 = recast<half2>(tzR);


    #pragma unroll
    for(int i = 0; i < size(tzR2); i++){
        // 括号产生fma
        tzR2(i) = a2 * txR2(i) + (b2 * tyR2(i) + c2);
    }

    auto tzRx = recast<half>(tzR2);
    copy(tzRx, tzr);
}
    

int main(){
    constexpr unsigned N_DATA{256};

    thrust::host_vector<half> h_x(N_DATA);
    for(int i = 0; i < N_DATA; i++){
        h_x[i] = static_cast<half>(i);
    }
    thrust::device_vector<half> d_x = h_x;

    thrust::host_vector<half> h_y(N_DATA);
    for(int i = 0; i < N_DATA; i++){
        h_y[i] = static_cast<half>(i);
    }
    thrust::device_vector<half> d_y = h_y;

    thrust::host_vector<half> h_z(N_DATA);
    for(int i = 0; i < N_DATA; i++){
        h_z[i] = static_cast<half>(-1);
    }
    thrust::device_vector<half> d_z = h_z;

    dim3 BLK_SIZE = {128};
    dim3 GRD_SIZE = {1};

    vec_add<<<GRD_SIZE, BLK_SIZE>>>(d_z.data().get(), N_DATA, d_x.data().get(), 
                                    d_y.data().get(), (half)2., (half)1., (half)0.5);

    h_z = d_z;

    for(int i = 0; i < N_DATA; i++){
        std::cout << (float)h_z[i] << " ";
        if(i % 8 == 7)  std::cout << std::endl;
    }
}

// nvcc --generate-code arch=compute_80,code=sm_80 -I/usr/local/cuda/include -I/home/yujixuan/kaiPro/cute-gemm-main/3rd/cutlass/include -L/usr/local/cuda/lib64 -l cuda -l cublas -o vec_add vecAdd.cu