# 1 "../step5-saxpy-kernel.cu"
# 59 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime.h"
#pragma GCC diagnostic ignored "-Wunused-function"
# 35 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/exception" 3
#pragma GCC visibility push ( default )
# 144 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/exception" 3
#pragma GCC visibility pop
# 42 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/new" 3
#pragma GCC visibility push ( default )
# 110 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/new" 3
#pragma GCC visibility pop
# 1425 "/usr/local/cuda-7.5/bin/..//include/driver_types.h"
struct CUstream_st;
# 206 "/usr/include/libio.h" 3
enum __codecvt_result {

__codecvt_ok,
__codecvt_partial,
__codecvt_error,
__codecvt_noconv};
# 199 "/usr/include/math.h" 3
enum _ZUt_ {
FP_NAN,

FP_INFINITE,

FP_ZERO,

FP_SUBNORMAL,

FP_NORMAL};
# 292 "/usr/include/math.h" 3
enum _LIB_VERSION_TYPE {
_IEEE_ = (-1),
_SVID_,
_XOPEN_,
_POSIX_,
_ISOC_};
# 124 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_voidIvEUt_E { _ZNSt9__is_voidIvE7__valueE = 1};
# 144 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIbEUt_E { _ZNSt12__is_integerIbE7__valueE = 1};
# 151 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIcEUt_E { _ZNSt12__is_integerIcE7__valueE = 1};
# 158 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIaEUt_E { _ZNSt12__is_integerIaE7__valueE = 1};
# 165 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIhEUt_E { _ZNSt12__is_integerIhE7__valueE = 1};
# 173 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIwEUt_E { _ZNSt12__is_integerIwE7__valueE = 1};
# 197 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIsEUt_E { _ZNSt12__is_integerIsE7__valueE = 1};
# 204 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerItEUt_E { _ZNSt12__is_integerItE7__valueE = 1};
# 211 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIiEUt_E { _ZNSt12__is_integerIiE7__valueE = 1};
# 218 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIjEUt_E { _ZNSt12__is_integerIjE7__valueE = 1};
# 225 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIlEUt_E { _ZNSt12__is_integerIlE7__valueE = 1};
# 232 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerImEUt_E { _ZNSt12__is_integerImE7__valueE = 1};
# 239 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIxEUt_E { _ZNSt12__is_integerIxE7__valueE = 1};
# 246 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt12__is_integerIyEUt_E { _ZNSt12__is_integerIyE7__valueE = 1};
# 264 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIfEUt_E { _ZNSt13__is_floatingIfE7__valueE = 1};
# 271 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIdEUt_E { _ZNSt13__is_floatingIdE7__valueE = 1};
# 278 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt13__is_floatingIeEUt_E { _ZNSt13__is_floatingIeE7__valueE = 1};
# 354 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIcEUt_E { _ZNSt9__is_charIcE7__valueE = 1};
# 362 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_charIwEUt_E { _ZNSt9__is_charIwE7__valueE = 1};
# 377 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIcEUt_E { _ZNSt9__is_byteIcE7__valueE = 1};
# 384 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIaEUt_E { _ZNSt9__is_byteIaE7__valueE = 1};
# 391 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/bits/cpp_type_traits.h" 3
enum _ZNSt9__is_byteIhEUt_E { _ZNSt9__is_byteIhE7__valueE = 1};
# 211 "/usr/lib/gcc/x86_64-redhat-linux/4.4.7/include/stddef.h" 3
typedef unsigned long size_t;
#include "crt/host_runtime.h"
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 2782 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaConfigureCall(struct dim3, struct dim3, size_t, struct CUstream_st *);
# 2964 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMalloc(void **, size_t);
# 3101 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaFree(void *);
# 3999 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);
# 4800 "/usr/local/cuda-7.5/bin/..//include/cuda_runtime_api.h"
extern enum cudaError cudaMemset(void *, int, size_t);
# 5 "../step5-saxpy-kernel.cu"
extern void _Z5saxpyifPfS_(int, float, float *, float *);
# 19 "../step5-saxpy-kernel.cu"
extern void _Z15saxpy_line6_gpuifPfS_(int, float, float *, float *);
# 45 "../step5-saxpy-kernel.cu"
extern int main(void);
extern int __cudaSetupArgSimple();
extern int __cudaLaunch();
extern void __nv_dummy_param_ref();
extern void __nv_save_fatbinhandle_for_managed_rt();
extern int __cudaRegisterEntry();
extern int __cudaRegisterBinary();
static void __sti___26_step5_saxpy_kernel_cpp1_ii_0cd90e4b(void) __attribute__((__constructor__));
# 5 "../step5-saxpy-kernel.cu"
void _Z5saxpyifPfS_( int n,  float a,  float *x,  float *y) {  {
 int i;
# 6 "../step5-saxpy-kernel.cu"
i = 0; for (; (i < n); i++)
{
(y[i]) = ((a * (x[i])) + (y[i]));
} }
return;
}
# 19 "../step5-saxpy-kernel.cu"
void _Z15saxpy_line6_gpuifPfS_( int n,  float a,  float *x,  float *y) {
 float *__cuda_local_var_41951_10_non_const_x_gpu;
 float *__cuda_local_var_41952_10_non_const_y_gpu;
 int __cuda_local_var_41953_7_non_const_size;
# 33 "../step5-saxpy-kernel.cu"
 struct dim3 __cuda_local_var_41964_8_non_const_bs;
 struct dim3 __cuda_local_var_41965_8_non_const_ts;
# 22 "../step5-saxpy-kernel.cu"
__cuda_local_var_41953_7_non_const_size = 16777216;
cudaMalloc(((void **)(&__cuda_local_var_41951_10_non_const_x_gpu)), ((size_t)__cuda_local_var_41953_7_non_const_size));
cudaMalloc(((void **)(&__cuda_local_var_41952_10_non_const_y_gpu)), ((size_t)__cuda_local_var_41953_7_non_const_size));

cudaMemset(((void *)__cuda_local_var_41951_10_non_const_x_gpu), 0, ((size_t)__cuda_local_var_41953_7_non_const_size));
cudaMemset(((void *)__cuda_local_var_41952_10_non_const_y_gpu), 0, ((size_t)__cuda_local_var_41953_7_non_const_size));

cudaMemcpy(((void *)__cuda_local_var_41951_10_non_const_x_gpu), ((const void *)x), ((size_t)__cuda_local_var_41953_7_non_const_size), cudaMemcpyHostToDevice);
cudaMemcpy(((void *)__cuda_local_var_41952_10_non_const_y_gpu), ((const void *)y), ((size_t)__cuda_local_var_41953_7_non_const_size), cudaMemcpyHostToDevice);


{
# 421 "/usr/local/cuda-7.5/bin/..//include/vector_types.h"
(__cuda_local_var_41964_8_non_const_bs.x) = 4096U; (__cuda_local_var_41964_8_non_const_bs.y) = 1U; (__cuda_local_var_41964_8_non_const_bs.z) = 1U;
# 33 "../step5-saxpy-kernel.cu"
}
{
# 421 "/usr/local/cuda-7.5/bin/..//include/vector_types.h"
(__cuda_local_var_41965_8_non_const_ts.x) = 1024U; (__cuda_local_var_41965_8_non_const_ts.y) = 1U; (__cuda_local_var_41965_8_non_const_ts.z) = 1U;
# 34 "../step5-saxpy-kernel.cu"
}
(cudaConfigureCall(__cuda_local_var_41964_8_non_const_bs, __cuda_local_var_41965_8_non_const_ts, 0UL, ((struct CUstream_st *)0LL))) ? ((void)0) : (__device_stub__Z18saxpy_line6_kernelifPfS_(n, a, __cuda_local_var_41951_10_non_const_x_gpu, __cuda_local_var_41952_10_non_const_y_gpu));

cudaMemcpy(((void *)y), ((const void *)__cuda_local_var_41952_10_non_const_y_gpu), ((size_t)__cuda_local_var_41953_7_non_const_size), cudaMemcpyDeviceToHost);

cudaFree(((void *)__cuda_local_var_41951_10_non_const_x_gpu));
cudaFree(((void *)__cuda_local_var_41952_10_non_const_y_gpu));

return;
}

int main(void) {

 float *__cuda_local_var_41978_12_non_const_x;
# 47 "../step5-saxpy-kernel.cu"
 float *__cuda_local_var_41978_16_non_const_y;
 float __cuda_local_var_41979_11_non_const_a;
 int __cuda_local_var_41980_9_non_const_size;
# 49 "../step5-saxpy-kernel.cu"
__cuda_local_var_41980_9_non_const_size = 16777216;
__cuda_local_var_41978_12_non_const_x = ((float *)(malloc(((size_t)__cuda_local_var_41980_9_non_const_size))));
__cuda_local_var_41978_16_non_const_y = ((float *)(malloc(((size_t)__cuda_local_var_41980_9_non_const_size))));

__cuda_local_var_41979_11_non_const_a = (3.0F); {


 int i;
# 56 "../step5-saxpy-kernel.cu"
i = 0; for (; (i < 4194304); i++) {
(__cuda_local_var_41978_12_non_const_x[i]) = (2.0F);
(__cuda_local_var_41978_16_non_const_y[i]) = (0.0F);
} }

printf(((const char *)" data\n")); {
 int i;
# 62 "../step5-saxpy-kernel.cu"
i = 0; for (; (i < 5); ++i) { printf(((const char *)"y[%d] = %f, "), i, ((double)(__cuda_local_var_41978_16_non_const_y[i]))); } }
printf(((const char *)"\n"));

_Z15saxpy_line6_gpuifPfS_(4194304, __cuda_local_var_41979_11_non_const_a, __cuda_local_var_41978_12_non_const_x, __cuda_local_var_41978_16_non_const_y);

printf(((const char *)" result\n")); {
 int i;
# 68 "../step5-saxpy-kernel.cu"
i = 0; for (; (i < 5); ++i) { printf(((const char *)"y[%d] = %f, "), i, ((double)(__cuda_local_var_41978_16_non_const_y[i]))); } }
printf(((const char *)"\n"));

free(((void *)__cuda_local_var_41978_12_non_const_x));
free(((void *)__cuda_local_var_41978_16_non_const_y));

return 0;
}
static void __sti___26_step5_saxpy_kernel_cpp1_ii_0cd90e4b(void) {   }

#include "step5-saxpy-kernel.cudafe1.stub.c"
