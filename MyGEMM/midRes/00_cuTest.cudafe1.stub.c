#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "00_cuTest.fatbin.c"
extern void __device_stub__Z6vecAddPfS_S_(float *, float *, float *);
extern void __device_stub__Z9SGEMM_v26PfS_S_(float *, float *, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z6vecAddPfS_S_(float *__par0, float *__par1, float *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(float *, float *, float *))vecAdd)));}
# 19 "../Common.cuh"
void vecAdd( float *__cuda_0,float *__cuda_1,float *__cuda_2)
# 19 "../Common.cuh"
{__device_stub__Z6vecAddPfS_S_( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "midRes/00_cuTest.cudafe1.stub.c"
void __device_stub__Z9SGEMM_v26PfS_S_( float *__par0,  float *__par1,  float *__par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, float *))SGEMM_v26))); }
# 79 "00_cuTest.cu"
void SGEMM_v26( float *__cuda_0,float *__cuda_1,float *__cuda_2)
# 79 "00_cuTest.cu"
{__device_stub__Z9SGEMM_v26PfS_S_( __cuda_0,__cuda_1,__cuda_2);
# 250 "00_cuTest.cu"
}
# 1 "midRes/00_cuTest.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T6) {  __nv_dummy_param_ref(__T6); __nv_save_fatbinhandle_for_managed_rt(__T6); __cudaRegisterEntry(__T6, ((void ( *)(float *, float *, float *))SGEMM_v26), _Z9SGEMM_v26PfS_S_, (-1)); __cudaRegisterEntry(__T6, ((void ( *)(float *, float *, float *))vecAdd), _Z6vecAddPfS_S_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
