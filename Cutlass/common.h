#pragma once

#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s: %d, ", __FILE__, __LINE__);                      \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }


template<typename T>
__host__ void CpuInitializeMatrix(T *matrix, int rows, int columns, int data_case)
{
    std::default_random_engine e;
    e.seed(time(0));
    std::uniform_real_distribution<float> u(-5.0, 5.0);

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < columns; j++)
        {
            int offset = i * columns + j;
            switch (data_case)
            {
            case 0:
                matrix[offset] = 1.;
                break;
            case 1:
                matrix[offset] = offset;
                break;
            case 2:
                matrix[offset] = u(e);
                break;
            default:
                matrix[offset] = 1.;
                break;
            }
        }
    }
}

// case0=1 case1=offset case2=random
template<typename T>
__host__ void CpuAllocateMatrix(T **matrix, int rows, int columns, int data_case = 0)
{
    size_t sizeof_matrix = sizeof(T) * rows * columns;
    CUDA_CHECK(cudaMallocHost((void **)matrix, sizeof_matrix));
    CpuInitializeMatrix(*matrix, rows, columns, data_case);
}

template<typename T>
__global__ void InitializeMatrix_kernel(
    T *matrix,
    int rows,
    int columns,
    int data_case)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns)
    {
        int offset = i * columns + j;
        float value = 1.;
        switch (data_case)
        {
        case 1:
            value = offset;
            break;
        case 2:
            int const k = 16807;
            int const m = 16;
            value = float((offset * k % m) - m / 2);
            break;
        default:
            break;
        }
        matrix[offset] = value;
    }
}

template<typename T>
cudaError_t GpuInitializeMatrix(T *matrix, int rows, int columns, int data_case)
{

    dim3 block(16, 16);
    dim3 grid(
        (rows + block.x - 1) / block.x,
        (columns + block.y - 1) / block.y);

    InitializeMatrix_kernel<T><<<grid, block>>>(matrix, rows, columns, data_case);
    cudaDeviceSynchronize();

    return cudaGetLastError();
}

// case0=1 case1=offset case2=random
template<typename T>
void GpuAllocateMatrix(T **matrix, int rows, int columns, int data_case = 0)
{
    size_t sizeof_matrix = sizeof(T) * rows * columns;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix));
    CUDA_CHECK(GpuInitializeMatrix(*matrix, rows, columns, data_case));
}
