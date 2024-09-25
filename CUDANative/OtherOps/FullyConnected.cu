#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdio>
#include "helper_cuda.h"
#include "hostptr.hpp"
#include "cuptr.hpp"

// 输入元素个数为m，每个元素的特征向量长度为n
#define m 512
#define n 128
// 隐藏层神经元个数为 k
#define k 512
// 那么权重矩阵就是 m * k


int main(){

}