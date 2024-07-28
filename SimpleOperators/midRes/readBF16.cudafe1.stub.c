#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "readBF16.fatbin.c"
extern void __device_stub__Z6vecAddPfS_S_(float *, float *, float *);
extern void __device_stub__Z8readBF16Pf(float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z6vecAddPfS_S_(float *__par0, float *__par1, float *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(float *, float *, float *))vecAdd)));}
# 19 "../Common.cuh"
void vecAdd( float *__cuda_0,float *__cuda_1,float *__cuda_2)
# 19 "../Common.cuh"
{__device_stub__Z6vecAddPfS_S_( __cuda_0,__cuda_1,__cuda_2);


}
# 1 "midRes/readBF16.cudafe1.stub.c"
void __device_stub__Z8readBF16Pf( float *__par0) {  __cudaLaunchPrologue(1); __cudaSetupArgSimple(__par0, 0UL); __cudaLaunch(((char *)((void ( *)(float *))readBF16))); }
# 12 "readBF16.cu"
void readBF16( float *__cuda_0)
# 12 "readBF16.cu"
{__device_stub__Z8readBF16Pf( __cuda_0);



}
# 1 "midRes/readBF16.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T4) {  __nv_dummy_param_ref(__T4); __nv_save_fatbinhandle_for_managed_rt(__T4); __cudaRegisterEntry(__T4, ((void ( *)(float *))readBF16), _Z8readBF16Pf, (-1)); __cudaRegisterEntry(__T4, ((void ( *)(float *, float *, float *))vecAdd), _Z6vecAddPfS_S_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
