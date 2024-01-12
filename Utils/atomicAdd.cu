#include "../Common.cuh"
__device__ double atomicAdd_double(double* address, double val)
{
 
    unsigned long long* address_as_ull = (unsigned long long*)address;  // 将地址强制转换为unsigned long long类型的指针
    unsigned long long old = *address_as_ull, assumed;  // old保存原始值，assumed用于CAS操作
    
    do
    {
        
        assumed = old;  // 在每次循环开始时，将assumed设置为old的值
        old = atomicCAS
        (
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );  // 使用atomicCAS()函数进行比较和交换操作
    } while (assumed != old);  // 如果assumed和old不相等，说明操作失败，需要继续循环            
    
    return __longlong_as_double(old);  // 将old转换为double类型并返回
}