#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "step5-saxpy-kernel.fatbin.c"
extern void __device_stub__Z18saxpy_line6_kernelifPfS_(int, float, float *, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_26_step5_saxpy_kernel_cpp1_ii_0cd90e4b(void) __attribute__((__constructor__));
void __device_stub__Z18saxpy_line6_kernelifPfS_(int __par0, float __par1, float *__par2, float *__par3){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__cudaSetupArgSimple(__par2, 8UL);__cudaSetupArgSimple(__par3, 16UL);__cudaLaunch(((char *)((void ( *)(int, float, float *, float *))saxpy_line6_kernel)));}
# 13 "../step5-saxpy-kernel.cu"
void saxpy_line6_kernel( int __cuda_0,float __cuda_1,float *__cuda_2,float *__cuda_3)
# 13 "../step5-saxpy-kernel.cu"
{__device_stub__Z18saxpy_line6_kernelifPfS_( __cuda_0,__cuda_1,__cuda_2,__cuda_3);



}
# 1 "step5-saxpy-kernel.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T20) {  __nv_dummy_param_ref(__T20); __nv_save_fatbinhandle_for_managed_rt(__T20); __cudaRegisterEntry(__T20, ((void ( *)(int, float, float *, float *))saxpy_line6_kernel), _Z18saxpy_line6_kernelifPfS_, (-1)); }
static void __sti____cudaRegisterAll_26_step5_saxpy_kernel_cpp1_ii_0cd90e4b(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }
