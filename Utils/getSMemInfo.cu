#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <ctime>
#include "../Common.cuh"

// 如何获得设备的各种信息
// 1. cudaGetDeviceProperties();
// 2. cudaDeviceGetxxx();
// 3. getCudaAttribute();

int main(){
    int smemSizePerSM, smemSizePerBlock;
    CHECK(cudaDeviceGetAttribute(&smemSizePerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CHECK(cudaDeviceGetAttribute(&smemSizePerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "每个SM的共享内存总量: " << smemSizePerSM << std::endl;
    std::cout << "每个block的共享内存总量: " << smemSizePerBlock << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "每个SM的寄存器总量：" << prop.regsPerMultiprocessor << std::endl;
    std::cout << "每个block的寄存器总量：" << prop.regsPerBlock << std::endl;


    // 好像是fixed 没法改
    CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    if(pConfig == cudaSharedMemBankSizeFourByte){
        std::cout << "共享内存Bank宽度为 4 字节" << std::endl;
    }
    else if(pConfig == cudaSharedMemBankSizeEightByte){
        std::cout << "共享内存Bank宽度为 8 字节" << std::endl;
    }
    else{
        std::cout << "其他？！" << std::endl;
    }
    // Ampere架构每个存储体每个时钟周期可以读取128位的数据

    int l1CacheForGlobal, l1CacheForLocal;
    cudaDeviceGetAttribute(&l1CacheForGlobal, cudaDevAttrGlobalL1CacheSupported, 0);
    cudaDeviceGetAttribute(&l1CacheForLocal, cudaDevAttrLocalL1CacheSupported, 0);
    if(l1CacheForGlobal > 0){
        std::cout << "L1 cache is enabled for Global!" << std::endl;
    }
    else{
        std::cout << "L1 cache is disabled for Global!!" << std::endl;
    }
    if(l1CacheForLocal > 0){
        std::cout << "L1 cache is enabled for Local!" << std::endl;
    }
    else{
        std::cout << "L1 cache is disabled for Local!!" << std::endl;
    }
    
    cudaFuncCache cacheConfig;
    CHECK(cudaDeviceGetCacheConfig(&cacheConfig));
    if(cacheConfig == cudaFuncCachePreferNone){
        std::cout << "在L1和共享内存间没有偏好" << std::endl;
    }
    else if(cacheConfig == cudaFuncCachePreferShared){
        std::cout << "PreferSharedMem" << std::endl;
    }
    else if(cacheConfig == cudaFuncCachePreferL1){
        std::cout << "PreferL1cache" << std::endl;
    }
    else{
        std::cout << "Prefer L1==SMem" << std::endl;
    }
    
}

// GA10x配置
// 128 KB L1 + 0 KB Shared Memory
// 120 KB L1 + 8 KB Shared Memory
// 112 KB L1 + 16 KB Shared Memory
// 96 KB L1 + 32 KB Shared Memory
// 64 KB L1 + 64 KB Shared Memory
// 28 KB L1 + 100 KB Shared Memory