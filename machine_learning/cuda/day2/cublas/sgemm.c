/* Includes, system */
#include <stdio.h>
#include <stdlib.h>


/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Matrix size */
// best
#define M  (4096) // 2048
#define N  (4096) // 1792
#define K  (4096) // 2048


/* Main */
int main(int argc, char **argv)
{

  int i;
  // for cuBLAS
  cublasStatus_t status;
  cublasHandle_t handle;
  float ms = 0.0;
  float s = 0.0;
  // for timer
  cudaEvent_t start, stop;

  float *h_A,*h_B, *h_C;
  float *d_A,*d_B, *d_C;

  float alpha = 1.0f;
  float beta = 0.0f;

  int nA = N * K;
  int nB = K * M;
  int nC = N * M;


  double num_gemm_op_real;
  double num_gemm_gop_real;

  unsigned long long  num_gemm_op_mul = (unsigned long long  )M*N*(K + 3);
  unsigned long long  num_gemm_op_add = (unsigned long long  )M*N*K;
  unsigned long long  num_gemm_op_total_seperate = (unsigned long long )num_gemm_op_add + num_gemm_op_mul;
  unsigned long long  num_gemm_op_total_detail = (unsigned long long  )M*K*(2 * N + 3);
  unsigned long long  num_gemm_op_total_simple = (unsigned long long )2 * M*K*N;
  unsigned long long  num_gemm_op = num_gemm_op_total_detail;

  double num_gemm_gop = (double)num_gemm_op * 1e-9;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //printf("CUDA error ??   %s \n", cudaGetErrorString(cudaGetLastError()));

  /* Initialize CUBLAS */
//  printf("simpleCUBLAS test running..\n");
//  printf(" matrix size (%d %d)X (%d %d) X (%d %d)  \n", N, K, K, M, N, M);
//  printf(" need memory %5.2f MB  ", ((float) sizeof(float) *  (N*K + K*M + N*M)) / (1024 * 1024));

//  printf("Host Malloc..\n");
  /* Allocate host memory for the matrices */
  h_A = (float *)malloc(nA * sizeof(h_A[0]));
  h_B = (float *)malloc(nB * sizeof(h_B[0]));
  h_C = (float *)malloc(nC * sizeof(h_C[0]));

  /* Fill the matrices with test data */
//  printf("data Init..\n");
  for (i = 0; i < nA; i++)  {        h_A[i] = rand() / (double)RAND_MAX;  }
  for (i = 0; i < nB; i++)  {        h_B[i] = rand() / (double)RAND_MAX;  }
  for (i = 0; i < nC; i++)  {        h_C[i] = rand() / (double)RAND_MAX;  }


  status = cublasCreate(&handle);
  //printf("CUDA error ??  %s \n",  cudaGetErrorString(cudaGetLastError()));

//  printf("GPU Malloc..\n");

  cudaMalloc((void **)&d_A, nA * sizeof(d_A[0]));
  cudaMalloc((void **)&d_B, nB * sizeof(d_B[0]));
  cudaMalloc((void **)&d_C, nC * sizeof(d_C[0]));


  //printf("CUDA error ??   %s \n",  cudaGetErrorString(cudaGetLastError()));

//  printf("Upload..\n");
  /* Initialize the device matrices with the host matrices */
  cublasSetVector(nA, sizeof(h_A[0]), h_A, 1, d_A, 1);
  cublasSetVector(nB, sizeof(h_B[0]), h_B, 1, d_B, 1);
  cublasSetVector(nC, sizeof(h_C[0]), h_C, 1, d_C, 1);

  /* Performs operation using cublas */
//  printf("GPU sgemm..\n");

  cudaEventRecord(start, 0);
  //printf("CUDA error  event record start  ?? %s \n",  cudaGetErrorString(cudaGetLastError()));
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);

//  printf(" total operation : \t %5.2f GFLOP   \n", num_dgemm_gop);
//  printf(" total time      : \t %5.2f ms\n", ms);
  s = ms *0.001;

  num_gemm_op_real = (double)num_gemm_op / s;
  num_gemm_gop_real = (double)num_gemm_op_real *1e-9;
  //printf(" real perf     : \t %5.2f FLOPS \n", rop );
  printf(" real perf       : \t %5.2f GFLOPS \n", num_gemm_gop_real);

  /* Read the result back */
//  printf("download..\n");
  status = cublasGetVector(nC, sizeof(h_C[0]), d_C, 1, h_C, 1);

  /* Memory clean up */
  printf("Host free..\n");
  free(h_A);
  free(h_B);
  free(h_C);

//  printf("GPU Malloc..\n");
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  //printf("CUDA error ??   %s \n",  cudaGetErrorString(cudaGetLastError()));

  /* Shutdown */
  status = cublasDestroy(handle);

//  printf("CUDA error ??  %s \n", cudaGetErrorString(cudaGetLastError()));

  return 0;
}

